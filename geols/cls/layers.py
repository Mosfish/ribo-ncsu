################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

import functools
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from typing import Optional
from torch_geometric.utils import softmax
import math  # [V7.1 修复] 导入 math 模块

#########################################################################

class GVPConvLayer(nn.Module):
    '''
    (解码器层)
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            autoregressive=False, 
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The current node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            if self.norm_first:
                dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            else:
                dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        
        if self.norm_first:
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

class GVPConv(MessagePassing):
    '''
    (解码器引擎)
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.contiguous().reshape(x_v.shape[0], x_v.shape[1] * 3),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.reshape(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.reshape(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)
        
#########################################################################
# [V7 新增] V7 长程模块
#########################################################################
class GVPDynamicProjection(nn.Module):
    '''
    V7 "Long-Range" 模块 (SE(3) 等变动态投影).
    
    实现了 "Compress" (N -> r) 和 "Broadcast" (N x r) 两个阶段。
    这是一个独立的模块，没有 MessagePassing，因为它执行的是全局操作。
    '''
    def __init__(self, node_dims, heads=4, n_anchors=32, drop_rate=0.1,
                 activations=(F.silu, torch.sigmoid), 
                 vector_gate=True):
        
        super(GVPDynamicProjection, self).__init__()
        
        self.si, self.vi = node_dims
        self.so, self.vo = node_dims # 输出维度与输入维度一致
        self.n_anchors = n_anchors # 这是 r
        self.heads = heads
        
        assert self.so % heads == 0 and self.vo % heads == 0, \
               "Output dimensions must be divisible by number of heads"
        
        self.head_s_dim = self.so // heads
        self.head_v_dim = self.vo // heads
        self.head_dims = (self.head_s_dim, self.head_v_dim)
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        # --- 1. "Compress" (N -> r) 阶段 ---
        # 投影权重 P_i，必须是 SE(3) 不变的 (输出 (n_anchors, 0))
        self.weight_proj = GVP(node_dims, (self.n_anchors, 0),
                               activations=(None, None))

        # --- 2. "Broadcast" (N x r) 阶段 ---
        # 独立的 Q, K, V 投影 (无边偏置)
        self.to_q_long = GVP_(node_dims, (self.so, self.vo))
        self.to_k_long = GVP_(node_dims, (self.so, self.vo))
        self.to_v_long = GVP_(node_dims, (self.so, self.vo))
        
        # 独立的 "长程注意力大脑"
        # 输入: 拼接的 Q_i 和 K_j (per-head)
        attn_in_dims = (2 * self.head_s_dim, 2 * self.head_v_dim)
        # 输出: 1 个标量 (energy) per head
        self.attn_gvp_long = GVP(attn_in_dims, (1, 0), 
                                 activations=(None, None))

        # [V7.1 BUG 6 修复] 添加 Attention Dropout
        self.dp_attn = nn.Dropout(drop_rate) 

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor` 
                  s: [N, n_conf, si], V: [N, n_conf, vi, 3]
        :return: out_long: GVP tuple (s, V)
                  s: [N, n_conf, so], V: [N, n_conf, vo, 3]
        '''
        s, v = x
        n_nodes, n_conf = s.shape[:2]
        
        # --- 1. "Compress" (N -> r) ---
        # 计算 SE(3) 不变权重 P_i
        # p_weights_s: [N, n_conf, n_anchors]
        p_weights_s = self.weight_proj(x) 
        
        # Softmax over N 节点 (dim=0) (锚点从节点 "吸收" 信息)
        p_weights = F.softmax(p_weights_s, dim=0)

        # 计算 r 个全局锚点 (加权平均)
        # s_global: [n_conf, n_anchors, si]
        s_global = torch.einsum('nca, ncs -> cas', p_weights, s)
        # v_global: [n_conf, n_anchors, vi, 3]
        v_global = torch.einsum('nca, ncvd -> cavd', p_weights, v)

        # 转换回 [n_anchors, n_conf, ...] 格式
        s_global = s_global.transpose(0, 1) # [r, n_conf, si]
        v_global = v_global.transpose(0, 1) # [r, n_conf, vi, 3]
        x_global = (s_global, v_global) # 这就是 r 个锚点

        # --- 2. "Broadcast" (N x r Cross-Attention) ---
        # 独立的 Q, K, V 投影
        q_long = self.to_q_long(x)        # Q 来自 N 个节点
        k_long = self.to_k_long(x_global) # K 来自 r 个锚点
        v_long = self.to_v_long(x_global) # V 来自 r 个锚点

        # --- 重塑以分离 Heads ---
        s_q, v_q = q_long
        s_k, v_k = k_long
        s_v, v_v = v_long

        r = self.n_anchors
        H = self.heads
        s_h, v_h = self.head_s_dim, self.head_v_dim

        q = (s_q.reshape(n_nodes, n_conf, H, s_h), v_q.reshape(n_nodes, n_conf, H, v_h, 3))
        k = (s_k.reshape(r, n_conf, H, s_h), v_k.reshape(r, n_conf, H, v_h, 3))
        v = (s_v.reshape(r, n_conf, H, s_h), v_v.reshape(r, n_conf, H, v_h, 3))

        # --- 扩展维度以进行 N x r 交互 ---
        # q_s: [N, 1, n_conf, H, s_h], q_v: [N, 1, n_conf, H, v_h, 3]
        q_s_exp, q_v_exp = q[0].unsqueeze(1), q[1].unsqueeze(1)
        # k_s: [1, r, n_conf, H, s_h], k_v: [1, r, n_conf, H, v_h, 3]
        k_s_exp, k_v_exp = k[0].unsqueeze(0), k[1].unsqueeze(0)

        # [V7.2 CRITICAL BUG 修复] 
        # torch.cat 不会自动广播, 必须手动 .expand()
        # 目标形状: [N, r, n_conf, H, dims]
        q_s_expanded = q_s_exp.expand(n_nodes, r, n_conf, H, s_h)
        q_v_expanded = q_v_exp.expand(n_nodes, r, n_conf, H, v_h, 3)
        
        k_s_expanded = k_s_exp.expand(n_nodes, r, n_conf, H, s_h)
        k_v_expanded = k_v_exp.expand(n_nodes, r, n_conf, H, v_h, 3)
        
        # 现在 q_exp 和 k_exp 都是 [N, r, ...] 形状
        q_exp_expanded = (q_s_expanded, q_v_expanded)
        k_exp_expanded = (k_s_expanded, k_v_expanded)

        # --- 计算长程能量 (使用独立的 attn_gvp_long) ---
        
        # [V7.1 BUG 1 修复]
        # 现在可以安全地拼接了
        attn_input = tuple_cat(q_exp_expanded, k_exp_expanded, dim=-1)
        
        # [V7.1 BUG 2 确认] 
        # Claude 的 reshape 是错的, GVP 可以处理高维 batch, 此处保持原样
        energy_s = self.attn_gvp_long(attn_input) 
        energy_s = energy_s.squeeze(-1) # [N, r, n_conf, H]

        # [V7.1 BUG 7 修复] 添加温度缩放
        energy_s = energy_s / math.sqrt(self.head_s_dim)
        
        # --- Softmax over r 锚点 (dim=1) ---
        alpha = F.softmax(energy_s, dim=1)
        
        # [V7.1 BUG 6 修复] 添加 Attention Dropout
        alpha = self.dp_attn(alpha)

        # --- 加权聚合 V ---
        # alpha_s: [N, r, n_conf, H, 1]
        alpha_s = alpha.unsqueeze(-1)
        # alpha_v: [N, r, n_conf, H, 1, 1]
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        # v_exp: [1, r, n_conf, H, dims]
        v_exp = (v[0].unsqueeze(0), v[1].unsqueeze(0))
        
        # (alpha * v).sum(dim=1)
        # out_s: [N, n_conf, H, s_h]
        out_s = (alpha_s * v_exp[0]).sum(dim=1)
        # out_v: [N, n_conf, H, v_h, 3]
        out_v = (alpha_v * v_exp[1]).sum(dim=1)

        # --- 融合 Heads ---
        # out_s_flat: [N, n_conf, H * s_h] = [N, n_conf, so]
        out_s_flat = out_s.contiguous().reshape(n_nodes, n_conf, self.so)
        # out_v_flat: [N, n_conf, H * v_h, 3] = [N, n_conf, vo, 3]
        out_v_flat = out_v.contiguous().reshape(n_nodes, n_conf, self.vo, 3)
        
        out_long = (out_s_flat, out_v_flat)
        return out_long


#########################################################################
# [V7 新增] V7 核心融合层
#########################################################################
class GeoLSTLayer(nn.Module):
    '''
    V7 "Geo-LST" 核心层 (几何长短时 Transformer).
    
    用门控融合机制 (Gated Fusion) 结合了:
    1. Short-Range: V6 GVPAttentionConv (GNN)
    2. Long-Range: V7 GVPDynamicProjection (Global)
    
    这个层被设计为 GVPAttentionConvLayer 的 "V7 升级版" 替代品。
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            heads=4, 
            n_anchors=32,
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=True,
        ):
        
        super(GeoLSTLayer, self).__init__()
        
        self.node_dims = node_dims
        
        # 1. 短程模块 (V6 SOTA GNN)
        self.short_module = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads, 
            drop_rate=drop_rate, # [V7.1 修复] 传递 drop_rate
            activations=activations, 
            vector_gate=vector_gate
        )
        
        # 2. 长程模块 (V7 Dynamic Projection)
        self.long_module = GVPDynamicProjection(
            node_dims, heads=heads, n_anchors=n_anchors,
            drop_rate=drop_rate, # [V7.1 修复] 传递 drop_rate
            activations=activations,
            vector_gate=vector_gate
        )
        
        # 3. 门控融合 (Gated Fusion) 逻辑
        # DualLN (双重归一化)
        self.ln_short = LayerNorm(node_dims)
        self.ln_long = LayerNorm(node_dims)
        
        # 门控投影 "大脑"
        # [V7.1 ISSUE 5 升级]
        # 门控大脑输出 so 个门, 而不是 1 个
        # --- [V7.4 新增：门控初始化] ---
        # Grok 建议：初始化 bias 为正数，使 gate 初始值 ≈ 1
        # 这使得模型在训练开始时 "偏向 Short"，依赖 V6 SOTA 信号
        # g = sigmoid(output + 3.0) ≈ 0.95
        self.gate_proj = nn.Linear(node_dims[0] * 2, node_dims[0]) 
        nn.init.constant_(self.gate_proj.bias, 1.0)

        # 4. 标准 FFN 和残差连接
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        # [V7.1 BUG 3 修复]
        # self.norm 只需要 1 个 FFN pre-norm
        self.norm = nn.ModuleList([LayerNorm(node_dims)]) # <--- 修复: 只需 1 个
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        
        self.residual = residual
        self.norm_first = norm_first # 确保坚持 Pre-LN 结构

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_in = x # 保存原始输入用于残差连接
        
        # [V7.1 BUG 3 修复] 
        # 移除 x_norm1 = self.norm[0](x)
        
        # --- DualLN (双重归一化) ---
        x_short_in = self.ln_short(x) # [V7.1 修复] 直接作用在 x 上
        x_long_in = self.ln_long(x)   # [V7.1 修复] 直接作用在 x 上
        
        # --- 1. 平行计算 (Short & Long) ---
        out_short = self.short_module(x_short_in, edge_index, edge_attr)
        out_long = self.long_module(x_long_in)

        # --- 2. 计算 "门" g (SE(3) 不变) ---
        s_short, v_short = out_short
        s_long, v_long = out_long
        
        # Gating "大脑" 只查看 SE(3) 不变的标量信息
        s_concat = torch.cat([s_short, s_long], dim=-1)
        
        # [V7.1 ISSUE 5 升级]
        # gate: [N, n_conf, so] (per-channel)
        gate = torch.sigmoid(self.gate_proj(s_concat))
        
        # --- 3. 门控融合 (Gated Fusion) ---
        # s_fused: [N, n_conf, so] (逐通道门控)
        s_fused = (gate * s_short) + ((1 - gate) * s_long)
        
        # [V7.1 ISSUE 5 升级]
        # 对于向量, 我们取标量门的平均值
        # gate_v: [N, n_conf, 1, 1]
        gate_v = gate.mean(dim=-1, keepdim=True).unsqueeze(-1)
        v_fused = (gate_v * v_short) + ((1 - gate_v) * v_long)
        
        x_fused = (s_fused, v_fused)
        
        # --- 4. 第一个残差连接 (Pre-LN) ---
        x_res = tuple_sum(x_in, self.dropout[0](x_fused))
        
        # --- 5. FFN Block (Pre-LN) ---
        # [V7.1 BUG 3 修复] 使用 self.norm[0]
        x_norm2 = self.norm[0](x_res) 
        x_ff = self.ff_func(x_norm2)
        x_out = tuple_sum(x_res, self.dropout[1](x_ff))
        
        return x_out

#########################################################################
# [V6 引擎] V6 GNN 核心 (被 V7 GeoLSTLayer 调用)
#########################################################################
class GVPAttentionConv(MessagePassing):
    '''
    GVP-Native Attention-based Message Passing (V6 Architecture)
    
    Combines GVP's SE(3)-equivariance with MHA, GAT, and TransformerConv
    (edge bias) concepts in a unified layer.
    
    - Q, K, V projections are GVP-native.
    - Attention energy is computed by a GVP-based 'head' (GAT-style).
    - Edge features (GVP tuples) are used to bias K and V (Transformer-style).
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 heads=4, drop_rate=0.1, # [V7.1 修复] 添加 drop_rate
                 activations=(F.silu, torch.sigmoid), 
                 vector_gate=True):
        
        super(GVPAttentionConv, self).__init__(aggr="add", node_dim=0)
        
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        self.heads = heads
        assert self.so % heads == 0 and self.vo % heads == 0, \
               "Output dimensions must be divisible by number of heads"
        
        self.head_s_dim = self.so // heads
        self.head_v_dim = self.vo // heads
        self.head_dims = (self.head_s_dim, self.head_v_dim)

        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)

        # 1. Q, K, V Projections (GVP-Native)
        self.to_q = GVP_(in_dims, (self.so, self.vo))
        self.to_k = GVP_(in_dims, (self.so, self.vo))
        self.to_v = GVP_(in_dims, (self.so, self.vo))

        # 2. Edge Bias Projections (TransformerConv idea)
        self.edge_proj_k = GVP_(edge_dims, (self.so, self.vo))
        self.edge_proj_v = GVP_(edge_dims, (self.so, self.vo))

        # 3. Attention Head (GAT idea)
        # Input: concatenated Q_i and K_j_biased (per-head)
        attn_in_dims = (2 * self.head_s_dim, 2 * self.head_v_dim)
        # Output: 1 scalar (energy) per head
        self.attn_gvp = GVP(attn_in_dims, (1, 0), 
                             activations=(None, None))

        # [V7.1 BUG 6 修复] 添加 Attention Dropout
        self.dp_attn = nn.Dropout(drop_rate) 

        # 4. Final Output Projection (Multi-head fusion)
        self.out_proj = GVP_((self.so, self.vo), (self.so, self.vo))
        

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor` 
                  s: [n_nodes, n_conf, si], V: [n_nodes, n_conf, vi, 3]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
                  s: [n_edges, n_conf, se], V: [n_edges, n_conf, ve, 3]
        '''
        x_s, x_v = x
        n_nodes, n_conf = x_s.shape[:2]

        # 1. Project Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 2. Project Edge Biases
        e_k = self.edge_proj_k(edge_attr)
        e_v = self.edge_proj_v(edge_attr)
        
        # --- Flatten all tuples for MessagePassing ---
        # We follow the GVPConv pattern of flattening vectors for propagation
        # (s, v) -> (s, v_flat) -> cat(s, v_flat)
        
        def flatten_heads(t, n, n_conf):
            s, v = t
            # --- [V6.4] BUG 修复 ---
            # .reshape() requires contiguous tensor, but GVP output (v) is transposed.
            # .reshape() is robust and handles both contiguous and non-contiguous cases.
            s_flat = s.reshape(n, n_conf * self.so)
            v_flat = v.reshape(n, n_conf * self.vo * 3)
            # --- [V6.4] 修复结束 ---
            return torch.cat([s_flat, v_flat], dim=-1)

        q_flat = flatten_heads(q, n_nodes, n_conf)
        k_flat = flatten_heads(k, n_nodes, n_conf)
        v_flat = flatten_heads(v, n_nodes, n_conf)
        
        n_edges = edge_attr[0].shape[0]
        e_k_flat = flatten_heads(e_k, n_edges, n_conf)
        e_v_flat = flatten_heads(e_v, n_edges, n_conf)

        # 3. Propagate messages
        message_flat = self.propagate(
            edge_index, 
            q=q_flat, k=k_flat, v=v_flat,
            e_k=e_k_flat, e_v=e_v_flat,
            n_conf=n_conf
        )

        # 4. Un-flatten aggregated message
        s_dim_flat = n_conf * self.so
        s_out = message_flat[..., :s_dim_flat].reshape(n_nodes, n_conf, self.so)
        v_out = message_flat[..., s_dim_flat:].reshape(n_nodes, n_conf, self.vo, 3)
        out = (s_out, v_out)

        # 5. Final output projection (fusion)
        final_out = self.out_proj(out)
        return final_out

    def message(self, q_i, k_j, v_j, e_k, e_v, n_conf: int,
                index: torch.Tensor, ptr: Optional[torch.Tensor], 
                size_i: Optional[int]) -> torch.Tensor:

        # --- Un-flatten all inputs ---
        
        # Helper to un-flatten and reshape to heads
        def unflatten_heads(t_flat, n_conf):
            s_dim_flat = n_conf * self.so
            s = t_flat[..., :s_dim_flat].reshape(-1, n_conf, self.heads, self.head_s_dim)
            v = t_flat[..., s_dim_flat:].reshape(-1, n_conf, self.heads, self.head_v_dim, 3)
            return s, v

        q_i = unflatten_heads(q_i, n_conf)
        k_j = unflatten_heads(k_j, n_conf)
        v_j = unflatten_heads(v_j, n_conf)
        e_k = unflatten_heads(e_k, n_conf)
        e_v = unflatten_heads(e_v, n_conf)

        # 1. Apply Edge Bias (in GVP space)
        k_j_biased = tuple_sum(k_j, e_k)
        v_j_biased = tuple_sum(v_j, e_v)

        # 2. Compute Attention Energy (GAT-style)
        # attn_input: (s_cat, v_cat)
        attn_input = tuple_cat(q_i, k_j_biased)
        
        # self.attn_gvp maps (2*head_dims) -> (heads, 0)
        # energy_s shape: [E, n_conf, H, 1]
        energy_s = self.attn_gvp(attn_input)
        energy_scalar = energy_s.squeeze(-1) # [E, n_conf, H]

        # [V7.1 BUG 7 修复] 添加温度缩放
        energy_scalar = energy_scalar / math.sqrt(self.head_s_dim)

        # 3. Softmax
        # We need to softmax over neighbors j for each node i.
        # Reshape for softmax: [E, n_conf * H]
        energy_flat = energy_scalar.reshape(-1, n_conf * self.heads)
        alpha_flat = softmax(energy_flat, index, ptr=ptr, dim=0)
        
        # [V7.1 BUG 6 修复] 添加 Attention Dropout
        alpha_flat = self.dp_attn(alpha_flat) # 在 softmax 之后应用

        # Reshape back: [E, n_conf, H]
        alpha = alpha_flat.reshape(-1, n_conf, self.heads)

        # 4. Apply Attention to Biased Value
        # alpha_s: [E, n_conf, H, 1]
        alpha_s = alpha.unsqueeze(-1)
        # alpha_v: [E, n_conf, H, 1, 1]
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        v_s_biased, v_v_biased = v_j_biased
        msg_s = alpha_s * v_s_biased # [E, n_conf, H, s_h]
        msg_v = alpha_v * v_v_biased # [E, n_conf, H, v_h, 3]

        # --- Flatten message for output ---
        # msg_s: [E, n_conf * H * s_h] = [E, n_conf * so]
        msg_s_flat = msg_s.contiguous().reshape(msg_s.shape[0], -1)
        # msg_v: [E, n_conf * H * v_h * 3] = [E, n_conf * vo * 3]
        msg_v_flat = msg_v.contiguous().reshape(msg_v.shape[0], -1)

        return torch.cat([msg_s_flat, msg_v_flat], dim=-1)

########################################################################
#########################################################################
# GVP 基础模块 (保持不变)
#########################################################################

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s
    
#########################################################################

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    [V7.1 BUG 1 修复]
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when reshaped
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    
    # [V7.1 BUG 1 修复] 
    # 实现文档字符串的逻辑
    v_dim = dim
    if dim == -1 or dim == len(args[0][0].shape) - 1:
        v_dim = -2 # 当 s_cat 在最后一个维度, v_cat 在倒数第二个维度
    
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=v_dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().reshape(x.shape[0], nv, 3)
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = v.contiguous().reshape(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)

def _split_multi(x, ns, nv, n_conf=5):
    '''
    _split for multiple conformers
    '''
    s = x[..., :-3 * nv * n_conf].contiguous().reshape(x.shape[0], n_conf, ns)
    v = x[..., -3 * nv * n_conf:].contiguous().reshape(x.shape[0], n_conf, nv, 3)
    return s, v

def _merge_multi(s, v):
    '''
    _merge for multiple conformers
    '''
    # s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
    s = s.contiguous().reshape(s.shape[0], s.shape[1] * s.shape[2])
    # v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
    v = v.contiguous().reshape(v.shape[0], v.shape[1] * v.shape[2] * 3)
    return torch.cat([s, v], -1)