################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
# MODIFIED: Added HybridGVPLongShortLayer for deep fusion of
# local geometric message passing and an efficient long-short attention mechanism.
# VERSION 3: Final robust version incorporating all valid feedback.
################################################################

import functools
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

#########################################################################
# FINAL VERSION: 辅助函数
def _create_batch_mask(batch, dtype, device):
    """根据 torch_geometric 的 batch 向量创建块对角注意力掩码。"""
    if batch is None:
        return None # 返回None，在主函数中处理

    num_nodes = batch.size(0)
    same_graph = batch.unsqueeze(1) == batch.unsqueeze(0)
    
    mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    mask.masked_fill_(~same_graph, torch.finfo(dtype).min)
    return mask

def _create_window_mask(seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ 高效的向量化窗口掩码生成 """
    half = window_size // 2
    idx = torch.arange(seq_len, device=device)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask.masked_fill_(dist > half, torch.finfo(dtype).min)
    return mask

# FINAL VERSION: Long-Short Attention 模块
class LongShortAttention(nn.Module):
    """
    实现了Long-Short Transformer思想的注意力模块.
    - 短程: 滑动窗口自注意力
    - 长程: 动态投影
    - 融合: DualLN + 拼接注意力得分
    """
    def __init__(self, embed_dim, num_heads, window_size, proj_rank, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.proj_rank = proj_rank
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.d_proj_matrix_proj = nn.Linear(embed_dim, proj_rank)
        
        self.ln_q = nn.LayerNorm(embed_dim)
        # 采纳建议：为K和V使用独立的LN
        self.ln_k_short = nn.LayerNorm(embed_dim)
        self.ln_v_short = nn.LayerNorm(embed_dim)
        self.ln_k_long = nn.LayerNorm(embed_dim)
        self.ln_v_long = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, batch=None):
        bsz, seq_len, _ = x.shape
        
        x_norm = self.ln_q(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k_short, v_short = self.ln_k_short(k), self.ln_v_short(v)
        k_long, v_long = self.ln_k_long(k), self.ln_v_long(v)

        q_s = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_s = k_short.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = v_short.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        window_mask = _create_window_mask(seq_len, self.window_size, x.device, x.dtype)
        batch_mask = _create_batch_mask(batch, x.dtype, x.device)
        
        # 采纳建议：在主函数中处理None
        if batch_mask is not None:
            attn_mask = window_mask + batch_mask
        else:
            attn_mask = window_mask

        attn_scores_short = torch.matmul(q_s, k_s.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 在使用时添加广播维度
        attn_scores_short = attn_scores_short + attn_mask.unsqueeze(0)
        
        proj_matrix = F.softmax(self.d_proj_matrix_proj(k_long), dim=1).transpose(1, 2)
        k_compressed = torch.matmul(proj_matrix, k_long) 
        v_compressed = torch.matmul(proj_matrix, v_long)

        q_l = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_c = k_compressed.view(bsz, self.proj_rank, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = v_compressed.view(bsz, self.proj_rank, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores_long = torch.matmul(q_l, k_c.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attn_scores_fused = torch.cat([attn_scores_short, attn_scores_long], dim=-1)
        attn_probs = F.softmax(attn_scores_fused, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_probs_short = attn_probs[..., :seq_len]
        attn_probs_long = attn_probs[..., seq_len:]

        output_short = torch.matmul(attn_probs_short, v_s)
        output_long = torch.matmul(attn_probs_long, v_c)
        
        output = output_short + output_long
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        
        return self.out_proj(output)

# (PositionwiseFeedForward 类保持不变)
class PositionwiseFeedForward(nn.Module):
    ''' A two-layer Feed-Forward-Network. '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.silu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# FINAL VERSION: HybridGVPLongShortLayer
class HybridGVPLongShortLayer(nn.Module):
    '''
    将GVP-GNN与Long-Short Attention思想深度融合的混合层。
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            num_attn_head=4,
            ls_window_size=50,
            ls_proj_rank=32,
            n_message=3, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
        ):
        super(HybridGVPLongShortLayer, self).__init__()
        
        self.norm_gvp = LayerNorm(node_dims)
        self.gvp_conv = MultiGVPConv(node_dims, node_dims, edge_dims, n_message,
                                     aggr="mean", activations=activations, vector_gate=vector_gate)
        self.dropout_gvp = Dropout(drop_rate)
        
        self.norm_attn = nn.LayerNorm(node_dims[0])
        self.ls_attention = LongShortAttention(embed_dim=node_dims[0], num_heads=num_attn_head, 
                                               window_size=ls_window_size, proj_rank=ls_proj_rank,
                                               dropout=drop_rate)
        self.dropout_attn = nn.Dropout(drop_rate)
        
        self.norm_ffn = nn.LayerNorm(node_dims[0])
        self.ffn = PositionwiseFeedForward(d_in=node_dims[0], d_hid=node_dims[0] * 4, dropout=drop_rate)
        self.dropout_ffn = nn.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_attr, batch=None):
        
        x_res_gvp = x
        x_norm_gvp = self.norm_gvp(x)
        dh_local = self.gvp_conv(x_norm_gvp, edge_index, edge_attr)
        x_local = tuple_sum(x_res_gvp, self.dropout_gvp(dh_local))
        s_local, v_local = x_local

        s_res_attn = s_local
        s_norm_attn = self.norm_attn(s_local)
        
        n_nodes, n_conf, d_s = s_norm_attn.shape
        s_reshaped = s_norm_attn.permute(1, 0, 2)
        
        s_attn_out = self.ls_attention(s_reshaped, batch=batch)
        
        s_attn_out = s_attn_out.permute(1, 0, 2)
        s_after_attn = s_res_attn + self.dropout_attn(s_attn_out)

        s_res_ffn = s_after_attn
        s_norm_ffn = self.norm_ffn(s_after_attn)
        s_ffn_out = self.ffn(s_norm_ffn)
        s_final = s_res_ffn + self.dropout_ffn(s_ffn_out)
        
        v_final = v_local
        
        return (s_final, v_final)

class GVPConvLayer(nn.Module):
    '''
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

class MultiGVPConvLayer(nn.Module):
    '''
    GVPConvLayer for handling multiple conformations (encoder-only)
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(MultiGVPConvLayer, self).__init__()
        self.conv = MultiGVPConv(node_dims, node_dims, edge_dims, n_message,
                                 aggr="mean", activations=activations, vector_gate=vector_gate)
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

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
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

class MultiGVPConv(MessagePassing):
    '''
    GVPConv for handling multiple conformations
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(MultiGVPConv, self).__init__(aggr=aggr)
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
        n_conf = x_s.shape[1]
        
        # x_s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
        x_s = x_s.contiguous().view(x_s.shape[0], x_s.shape[1] * x_s.shape[2])        
        # x_v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
        x_v = x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * x_v.shape[2] * 3)
        
        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)
        
        return _split_multi(message, self.so, self.vo, n_conf)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        # [n_nodes, n_conf * d] -> [n_nodes, n_conf, d]
        s_i = s_i.view(s_i.shape[0], s_i.shape[1]//self.si, self.si)
        s_j = s_j.view(s_j.shape[0], s_j.shape[1]//self.si, self.si)
        # [n_nodes, n_conf * d * 3] -> [n_nodes, n_conf, d, 3]
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//(self.vi * 3), self.vi, 3)
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//(self.vi * 3), self.vi, 3)

        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge_multi(*message)

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
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

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
