################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
#
# V8 MODIFICATION:
# This file contains the complete V8 GeoLST architecture.
# 1. GVPAttentionConv (V5.1 SOTA Dot-Product) - Short-Range Module
# 2. GVPGraphormerAttention (V8) - Long-Range Global Attention (Graphormer-style)
# 3. GeoLSTLayerV8 (V8) - Fusion Layer with Gated Mechanism
# 4. GVPConvLayer / GVPConv - gRNAde baseline (used by decoder)
# 5. GVP base modules (GVP, LayerNorm, etc.)
################################################################

import functools
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from typing import Optional

# [V7-MOD] 导入 math 和 softmax
import math 
from torch_geometric.utils import softmax
#########################################################################

class GVPConvLayer(nn.Module):
    '''
    (gRNAde 原版解码器层)
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons.
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
    (gRNAde 原版 GNN 卷积)
    Graph convolution / message passing with Geometric Vector Perceptrons.
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
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)
        
#########################################################################
# [V7 短程模块 封装层] V7 Short-Range Wrapper (GVPAttentionConvLayer)
#########################################################################
class GVPAttentionConvLayer(nn.Module):
    '''
    (V7 短程模块封装)
    Full GVP Attention Layer (V5.1 SOTA / V7 Short-Range)
    
    Applies 'GVPAttentionConv' message passing, followed by
    residual updates and a pointwise feedforward network.
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            heads=4, 
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(GVPAttentionConvLayer, self).__init__()
        
        # Use GVPAttentionConv as the convolution module
        self.conv = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads, 
            drop_rate=drop_rate, # [V7-MOD] 传递 drop_rate
            activations=activations, 
            vector_gate=vector_gate
        )
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # Standard GVP FeedForward network
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

    def forward(self, x, edge_index, edge_attr):
        if self.norm_first:
            dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            dh = self.conv(x, edge_index, edge_attr)
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        return x

#########################################################################
# [V7 短程模块 核心] V7 Short-Range Core (GVPAttentionConv)
#########################################################################
class GVPAttentionConv(MessagePassing):
    '''
    (V7 短程模块核心)
    GVP-Native Attention-based Message Passing (V5.1 SOTA / V7 Short-Range)
    - Q, K, V projections are GVP-native.
    - Attention energy is computed by a GVP-native **Dot-Product**.
    - Edge features (GVP tuples) are used to bias K and V.
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 heads=4,
                 drop_rate=0.1, # [V7-MOD] 接收 drop_rate
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

        # Q, K, V Projections from GVP
        self.to_q = GVP_(in_dims, (self.so, self.vo))
        self.to_k = GVP_(in_dims, (self.so, self.vo))
        self.to_v = GVP_(in_dims, (self.so, self.vo))

        # Edge Bias Projections
        self.edge_proj_k = GVP_(edge_dims, (self.so, self.vo))
        self.edge_proj_v = GVP_(edge_dims, (self.so, self.vo))

        # [V7-MOD] 删除 GAT-style 的 attn_gvp
        
        # [V7-MOD] 添加 Attention Dropout
        self.dp_attn = nn.Dropout(drop_rate) 

        # Output Projection (Multi-head fusion)
        self.out_proj = GVP_((self.so, self.vo), (self.so, self.vo))
        

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        n_nodes, n_conf = x_s.shape[:2]

        # Project Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Project Edge Biases
        e_k = self.edge_proj_k(edge_attr)
        e_v = self.edge_proj_v(edge_attr)
        
        # --- Flatten all tuples for MessagePassing ---
        def flatten_heads(t, n, n_conf):
            s, v = t
            s_flat = s.contiguous().view(n, n_conf * self.so)
            v_flat = v.contiguous().view(n, n_conf * self.vo * 3)
            return torch.cat([s_flat, v_flat], dim=-1)

        q_flat = flatten_heads(q, n_nodes, n_conf)
        k_flat = flatten_heads(k, n_nodes, n_conf)
        v_flat = flatten_heads(v, n_nodes, n_conf)
        
        n_edges = edge_attr[0].shape[0]
        e_k_flat = flatten_heads(e_k, n_edges, n_conf)
        e_v_flat = flatten_heads(e_v, n_edges, n_conf)

        # Propagate messages
        message_flat = self.propagate(
            edge_index, 
            q=q_flat, k=k_flat, v=v_flat,
            e_k=e_k_flat, e_v=e_v_flat,
            n_conf=n_conf
        )

        # Un-flatten aggregated message
        s_dim_flat = n_conf * self.so
        s_out = message_flat[..., :s_dim_flat].view(n_nodes, n_conf, self.so)
        v_out = message_flat[..., s_dim_flat:].view(n_nodes, n_conf, self.vo, 3)
        out = (s_out, v_out)

        # Final output projection (fusion)
        final_out = self.out_proj(out)
        return final_out

    def message(self, q_i, k_j, v_j, e_k, e_v, n_conf: int,
                index: torch.Tensor, ptr: Optional[torch.Tensor], 
                size_i: Optional[int]) -> torch.Tensor:

        # --- Un-flatten all inputs ---
        def unflatten_heads(t_flat, n_conf):
            s_dim_flat = n_conf * self.so
            s = t_flat[..., :s_dim_flat].view(-1, n_conf, self.heads, self.head_s_dim)
            v = t_flat[..., s_dim_flat:].view(-1, n_conf, self.heads, self.head_v_dim, 3)
            return s, v

        q_i = unflatten_heads(q_i, n_conf)
        k_j = unflatten_heads(k_j, n_conf)
        v_j = unflatten_heads(v_j, n_conf)
        e_k = unflatten_heads(e_k, n_conf)
        e_v = unflatten_heads(e_v, n_conf)

        # Apply Edge Bias
        k_j_biased = tuple_sum(k_j, e_k)
        v_j_biased = tuple_sum(v_j, e_v)

        # --- [V7-MOD] 替换 GAT-style 为 GVP-Dot-Product ---
        
        # 2. Compute Attention Energy (Transformer-style Dot-Product)
        s_q, v_q = q_i
        s_k_biased, v_k_biased = k_j_biased

        s_dot = s_q * s_k_biased
        v_dot = (v_q * v_k_biased).sum(dim=-1) 

        s_energy = s_dot.sum(dim=-1) + v_dot.sum(dim=-1) 
        
        head_dim = self.head_s_dim + self.head_v_dim
        energy_scalar = s_energy / (math.sqrt(head_dim) if head_dim > 0 else 1.0)
        # ----------------------------------------------------

        # 3. Softmax
        energy_flat = energy_scalar.view(-1, n_conf * self.heads)
        alpha_flat = softmax(energy_flat, index, ptr=ptr, dim=0)
        
        # [V7-MOD] 应用 Attention Dropout (在 softmax 之后)
        alpha_flat = self.dp_attn(alpha_flat)

        alpha = alpha_flat.view(-1, n_conf, self.heads)

        # 4. Apply Attention to Biased Value
        alpha_s = alpha.unsqueeze(-1)
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        v_s_biased, v_v_biased = v_j_biased
        msg_s = alpha_s * v_s_biased 
        msg_v = alpha_v * v_v_biased 

        # --- Flatten message for output ---
        msg_s_flat = msg_s.contiguous().view(msg_s.shape[0], -1)
        msg_v_flat = msg_v.contiguous().view(msg_v.shape[0], -1)

        return torch.cat([msg_s_flat, msg_v_flat], dim=-1)

#########################################################################
# [V7 长程模块] V7 Long-Range Module (GVPDynamicProjection)
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
        self.so, self.vo = node_dims 
        self.n_anchors = n_anchors 
        self.heads = heads
        
        assert self.so % heads == 0 and self.vo % heads == 0, \
               "Output dimensions must be divisible by number of heads"
        
        self.head_s_dim = self.so // heads
        self.head_v_dim = self.vo // heads
        self.head_dims = (self.head_s_dim, self.head_v_dim)
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        # --- 1. "Compress" (N -> r) 阶段 ---
        self.weight_proj = GVP(node_dims, (self.n_anchors, 0),
                               activations=(None, None))

        # --- 2. "Broadcast" (N x r) 阶段 ---
        self.to_q_long = GVP_(node_dims, (self.so, self.vo))
        self.to_k_long = GVP_(node_dims, (self.so, self.vo))
        self.to_v_long = GVP_(node_dims, (self.so, self.vo))
        
        # [V7-MOD] 长程注意力也使用 GVP-Dot-Product
        
        # [V7-MOD] 添加 Attention Dropout
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
        p_weights_s = self.weight_proj(x) 
        p_weights = F.softmax(p_weights_s, dim=0)

        s_global = torch.einsum('nca, ncs -> cas', p_weights, s)
        v_global = torch.einsum('nca, ncvd -> cavd', p_weights, v)

        s_global = s_global.transpose(0, 1) # [r, n_conf, si]
        v_global = v_global.transpose(0, 1) # [r, n_conf, vi, 3]
        x_global = (s_global, v_global) # 这就是 r 个锚点

        # --- 2. "Broadcast" (N x r Cross-Attention) ---
        q_long = self.to_q_long(x)        # Q 来自 N 个节点
        k_long = self.to_k_long(x_global) # K 来自 r 个锚点
        v_long = self.to_v_long(x_global) # V 来自 r 个锚点

        # --- 重塑以分离 Heads ---
        s_q, v_q = q_long
        s_k, v_k = k_long
        s_val, v_val = v_long  # 改名避免与下面的 val 变量混淆

        r = self.n_anchors
        H = self.heads
        s_h, v_h = self.head_s_dim, self.head_v_dim

        q = (s_q.contiguous().view(n_nodes, n_conf, H, s_h), v_q.contiguous().view(n_nodes, n_conf, H, v_h, 3))
        k = (s_k.contiguous().view(r, n_conf, H, s_h), v_k.contiguous().view(r, n_conf, H, v_h, 3))
        val = (s_val.contiguous().view(r, n_conf, H, s_h), v_val.contiguous().view(r, n_conf, H, v_h, 3))

        # --- [V7-MOD] GVP-Dot-Product 计算能量 ---
        # q_s: [N, n_conf, H, s_h], q_v: [N, n_conf, H, v_h, 3]
        # k_s: [r, n_conf, H, s_h], k_v: [r, n_conf, H, v_h, 3]
        
        # einsum 计算 N x r 的点积 (s_dot)
        # (n c h s), (r c h s) -> (n r c h)
        s_dot = torch.einsum('nchs, rchs -> nrch', q[0], k[0])
        
        # einsum 计算 N x r 的点积 (v_dot)
        # (n c h v d), (r c h v d) -> (n r c h)
        v_dot = torch.einsum('nchvd, rchvd -> nrch', q[1], k[1])

        # 聚合能量并缩放
        s_energy = s_dot + v_dot # [N, r, n_conf, H]
        head_dim = self.head_s_dim + self.head_v_dim
        energy_scalar = s_energy / (math.sqrt(head_dim) if head_dim > 0 else 1.0)
        # ----------------------------------------

        # --- Softmax over r 锚点 (dim=1) ---
        alpha = F.softmax(energy_scalar, dim=1)
        alpha = self.dp_attn(alpha) # 应用 Dropout

        # --- 加权聚合 V ---
        alpha_s = alpha.unsqueeze(-1)
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        # out_s: [N, n_conf, H, s_h]
        out_s = (alpha_s * val[0].unsqueeze(0)).sum(dim=1)
        # out_v: [N, n_conf, H, v_h, 3]
        out_v = (alpha_v * val[1].unsqueeze(0)).sum(dim=1)

        # --- 融合 Heads ---
        out_s_flat = out_s.contiguous().reshape(n_nodes, n_conf, self.so)
        out_v_flat = out_v.contiguous().reshape(n_nodes, n_conf, self.vo, 3)
        
        out_long = (out_s_flat, out_v_flat)
        return out_long


#########################################################################
# [V8 长程模块] V8 Long-Range Module (GVPGraphormerAttention)
#########################################################################
class GVPGraphormerAttention(nn.Module):
    '''
    V8 "Long-Range" 模块 (Graphormer-style 全局注意力).

    核心思想 (来自 Graphormer 论文):
    1. 全局注意力: 每个节点关注所有其他节点
    2. 空间编码: 使用最短路径距离作为注意力偏置
    3. 边编码: 使用边特征作为注意力偏置
    4. SE(3) 等变: 使用 GVP-Dot-Product
    '''
    def __init__(self, node_dims, edge_dims, heads=4, drop_rate=0.1,
                 max_path_distance=5,
                 activations=(F.silu, torch.sigmoid),
                 vector_gate=True):

        super(GVPGraphormerAttention, self).__init__()

        self.si, self.vi = node_dims
        self.so, self.vo = node_dims
        self.se, self.ve = edge_dims
        self.heads = heads
        self.max_path_distance = max_path_distance

        assert self.so % heads == 0 and self.vo % heads == 0, \
               "Output dimensions must be divisible by number of heads"

        self.head_s_dim = self.so // heads
        self.head_v_dim = self.vo // heads

        GVP_ = functools.partial(GVP,
                activations=activations, vector_gate=vector_gate)

        # Q/K/V 投影
        self.to_q = GVP_(node_dims, (self.so, self.vo))
        self.to_k = GVP_(node_dims, (self.so, self.vo))
        self.to_v = GVP_(node_dims, (self.so, self.vo))

        # [Graphormer] 空间编码: 最短路径距离的嵌入
        # 为每个距离学习一个标量偏置 (每个头独立)
        self.spatial_encoding = nn.Embedding(max_path_distance + 1, heads)
        nn.init.normal_(self.spatial_encoding.weight, std=0.02)

        # [Graphormer] 边编码: 边特征投影到注意力偏置
        # 输出维度: heads (每个头一个标量偏置)
        self.edge_encoder = GVP(edge_dims, (heads, 0),
                                activations=(None, None))

        # Attention Dropout
        self.dp_attn = nn.Dropout(drop_rate)

        # 输出投影
        self.out_proj = GVP_(node_dims, node_dims)

    def forward(self, x, edge_index, edge_attr, spatial_encoding=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
                  s: [N, n_conf, si], V: [N, n_conf, vi, 3]
        :param edge_index: [2, E] 边索引
        :param edge_attr: tuple (s, V) 边特征
        :param spatial_encoding: [N, N] 最短路径距离矩阵 (可选)
        :return: out: GVP tuple (s, V)
        '''
        s, v = x
        n_nodes, n_conf = s.shape[:2]

        # --- 1. Q/K/V 投影 ---
        q = self.to_q(x)  # [N, n_conf, so], [N, n_conf, vo, 3]
        k = self.to_k(x)
        val = self.to_v(x)

        # --- 2. 重塑为多头 ---
        # q_s: [N, n_conf, H, s_h]
        q_s = q[0].contiguous().view(n_nodes, n_conf, self.heads, self.head_s_dim)
        q_v = q[1].contiguous().view(n_nodes, n_conf, self.heads, self.head_v_dim, 3)

        k_s = k[0].contiguous().view(n_nodes, n_conf, self.heads, self.head_s_dim)
        k_v = k[1].contiguous().view(n_nodes, n_conf, self.heads, self.head_v_dim, 3)

        val_s = val[0].contiguous().view(n_nodes, n_conf, self.heads, self.head_s_dim)
        val_v = val[1].contiguous().view(n_nodes, n_conf, self.heads, self.head_v_dim, 3)

        # --- 3. GVP-Dot-Product 计算注意力能量 (全局 N x N) ---
        # s_dot: [N, N, n_conf, H]
        s_dot = torch.einsum('ichs, jchs -> ijch', q_s, k_s)
        v_dot = torch.einsum('ichvd, jchvd -> ijch', q_v, k_v)

        energy = s_dot + v_dot  # [N, N, n_conf, H]

        # 缩放
        head_dim = self.head_s_dim + self.head_v_dim
        energy = energy / (math.sqrt(head_dim) if head_dim > 0 else 1.0)

        # --- 4. [Graphormer] 添加空间编码偏置 ---
        if spatial_encoding is not None:
            # spatial_encoding: [N, N] 最短路径距离
            # 限制最大距离
            spatial_encoding_clamped = torch.clamp(
                spatial_encoding, max=self.max_path_distance
            )
            # 获取嵌入: [N, N, H]
            spatial_bias = self.spatial_encoding(spatial_encoding_clamped)
            # 广播到 [N, N, n_conf, H]
            spatial_bias = spatial_bias.unsqueeze(2)  # [N, N, 1, H]
            energy = energy + spatial_bias

        # --- 5. [Graphormer] 添加边编码偏置 ---
        # 构建边偏置矩阵 [N, N, n_conf, H]
        edge_bias = torch.zeros(n_nodes, n_nodes, n_conf, self.heads,
                               device=s.device, dtype=s.dtype)

        if edge_attr is not None and edge_index.shape[1] > 0:
            # 投影边特征到标量偏置
            # [V8-FIX] GVP 输出是一个元组 (s, v)，我们只取标量部分 s
            edge_bias_values = self.edge_encoder(edge_attr)[0]  # [E, n_conf, heads]

            # 如果 edge_bias_values 是 [E, heads]，需要扩展到 [E, n_conf, heads]
            if edge_bias_values.dim() == 2:
                edge_bias_values = edge_bias_values.unsqueeze(1).expand(-1, n_conf, -1)

            # 填充到矩阵中
            src, dst = edge_index
            edge_bias[src, dst] = edge_bias_values
            # 对称化 (无向图)
            edge_bias[dst, src] = edge_bias_values

        # edge_bias 已经是 [N, N, n_conf, H]
        energy = energy + edge_bias

        # --- 6. Softmax (over all N nodes) ---
        # 重塑为 [N, N*n_conf*H] 以便使用 softmax
        energy_flat = energy.reshape(n_nodes, -1)  # [N, N*n_conf*H]

        # 对每个查询节点，在所有键节点上做 softmax
        alpha_flat = F.softmax(energy_flat, dim=1)  # [N, N*n_conf*H]
        alpha_flat = self.dp_attn(alpha_flat)

        # 重塑回 [N, N, n_conf, H]
        alpha = alpha_flat.reshape(n_nodes, n_nodes, n_conf, self.heads)

        # --- 7. 加权聚合 Value ---
        # alpha: [N, N, n_conf, H]
        # val_s: [N, n_conf, H, s_h]
        # 需要: [N(query), N(key), n_conf, H] x [N(key), n_conf, H, s_h] -> [N(query), n_conf, H, s_h]

        alpha_s = alpha.unsqueeze(-1)  # [N, N, n_conf, H, 1]
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)  # [N, N, n_conf, H, 1, 1]

        # 广播 val 到 [N(query), N(key), n_conf, H, s_h]
        val_s_expanded = val_s.unsqueeze(0)  # [1, N, n_conf, H, s_h]
        val_v_expanded = val_v.unsqueeze(0)  # [1, N, n_conf, H, v_h, 3]

        # 加权求和
        out_s = (alpha_s * val_s_expanded).sum(dim=1)  # [N, n_conf, H, s_h]
        out_v = (alpha_v * val_v_expanded).sum(dim=1)  # [N, n_conf, H, v_h, 3]

        # --- 8. 融合多头 ---
        out_s_flat = out_s.contiguous().view(n_nodes, n_conf, self.so)
        out_v_flat = out_v.contiguous().view(n_nodes, n_conf, self.vo, 3)

        out = (out_s_flat, out_v_flat)

        # --- 9. 输出投影 ---
        out = self.out_proj(out)

        return out


#########################################################################
# [V7 核心融合层] V7 Core Fusion Layer (GeoLSTLayer)
#########################################################################
class GeoLSTLayer(nn.Module):
    '''
    V7 "Geo-LST" 核心层 (几何长短时 Transformer).
    
    融合了:
    1. Short-Range: V5.1 SOTA GVPAttentionConv (GNN, 局部, 边偏置)
    2. Long-Range: V7 GVPDynamicProjection (Global, 无视 edge_index)
    
    融合策略:
    - DualLN (尺度归一化) + 加法融合 (稳定)
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
        
        # 1. 短程模块 (V5.1 SOTA 模块, 来自本文件)
        self.short_module = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads, 
            drop_rate=drop_rate,
            activations=activations, 
            vector_gate=vector_gate
        )
        
        # 2. 长程模块 (V7 全局模块, 来自本文件)
        self.long_module = GVPDynamicProjection(
            node_dims, heads=heads, n_anchors=n_anchors,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate
        )
        
        # 3. 融合模块 (LST DualLN)
        self.ln_short = LayerNorm(node_dims)
        self.ln_long = LayerNorm(node_dims)
        
        # 4. 标准 FFN 和残差连接
        GVP_ = functools.partial(GVP,
                activations=activations, vector_gate=vector_gate)

        # [V7-MOD] 修复 Bug：Pre-LN 结构只需要 1 个 Norm (用于 FFN)
        self.norm = LayerNorm(node_dims)
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
        x_in = x  # 保存原始输入用于残差连接

        # --- 1. DualLN (LST 论文核心) ---
        # 直接从 x_in 开始，避免过度归一化
        x_short_in = self.ln_short(x_in)
        x_long_in = self.ln_long(x_in)

        # --- 2. 平行计算 (Short & Long) ---
        # 短程：SOTA GVP-Attention (V5.1)
        out_short = self.short_module(x_short_in, edge_index, edge_attr)

        # 长程：SE(3) Dynamic Projection (V7)
        out_long = self.long_module(x_long_in)

        # --- 3. 融合：加法聚合 (LST 论文图 1) ---
        x_fused = tuple_sum(out_short, out_long)

        # --- 4. 第一个残差连接 (Pre-LN) ---
        x_res_1 = tuple_sum(x_in, self.dropout[0](x_fused))

        # --- 5. FFN Block (Pre-LN) ---
        x_norm_2 = self.norm(x_res_1)  # 使用单个 Norm
        x_ff = self.ff_func(x_norm_2)

        # --- 6. 第二个残差连接 ---
        x_out = tuple_sum(x_res_1, self.dropout[1](x_ff))

        return x_out


#########################################################################
# [V8 核心融合层] V8 Core Fusion Layer (GeoLSTLayerV8)
#########################################################################
class GeoLSTLayerV8(nn.Module):
    '''
    V8 "Geo-LST" 核心层 (几何长短时 Transformer with Graphormer).

    融合了:
    1. Short-Range: V5.1 SOTA GVPAttentionConv (GNN, 局部, 边偏置)
    2. Long-Range: V8 GVPGraphormerAttention (Graphormer 全局注意力)

    融合策略:
    - DualLN (尺度归一化) + 加法融合
    '''
    def __init__(
            self,
            node_dims,
            edge_dims,
            heads=4,
            max_path_distance=5,
            n_feedforward=2,
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid),
            vector_gate=True,
            residual=True,
            norm_first=True,
        ):

        super(GeoLSTLayerV8, self).__init__()

        # 1. 短程模块 (V5.1 SOTA 模块)
        self.short_module = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate
        )

        # 2. 长程模块 (V8 Graphormer 全局注意力)
        self.long_module = GVPGraphormerAttention(
            node_dims, edge_dims, heads=heads,
            max_path_distance=max_path_distance,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate
        )

        # 3. 融合模块 (DualLN)
        self.ln_short = LayerNorm(node_dims)
        self.ln_long = LayerNorm(node_dims)

        # 4. 标准 FFN 和残差连接
        GVP_ = functools.partial(GVP,
                activations=activations, vector_gate=vector_gate)

        self.norm = LayerNorm(node_dims)
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

    def forward(self, x, edge_index, edge_attr, spatial_encoding=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param spatial_encoding: [N, N] 最短路径距离矩阵 (可选)
        '''
        x_in = x  # 保存原始输入用于残差连接

        # --- 1. DualLN ---
        x_short_in = self.ln_short(x_in)
        x_long_in = self.ln_long(x_in)

        # --- 2. 平行计算 (Short & Long) ---
        # 短程：GVP-Attention (局部)
        out_short = self.short_module(x_short_in, edge_index, edge_attr)

        # 长程：Graphormer 全局注意力
        out_long = self.long_module(x_long_in, edge_index, edge_attr, spatial_encoding)

        # --- 3. 融合：加法聚合 ---
        x_fused = tuple_sum(out_short, out_long)

        # --- 4. 第一个残差连接 (Pre-LN) ---
        x_res_1 = tuple_sum(x_in, self.dropout[0](x_fused))

        # --- 5. FFN Block (Pre-LN) ---
        x_norm_2 = self.norm(x_res_1)
        x_ff = self.ff_func(x_norm_2)

        # --- 6. 第二个残差连接 ---
        x_out = tuple_sum(x_res_1, self.dropout[1](x_ff))

        return x_out


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
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)

def _split_multi(x, ns, nv, n_conf=5):
    '''
    _split for multiple conformers
    '''
    s = x[..., :-3 * nv * n_conf].contiguous().view(x.shape[0], n_conf, ns)
    v = x[..., -3 * nv * n_conf:].contiguous().view(x.shape[0], n_conf, nv, 3)
    return s, v

def _merge_multi(s, v):
    '''
    _merge for multiple conformers
    '''
    # s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
    s = s.contiguous().view(s.shape[0], s.shape[1] * s.shape[2])
    # v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
    v = v.contiguous().view(v.shape[0], v.shape[1] * v.shape[2] * 3)
    return torch.cat([s, v], -1)