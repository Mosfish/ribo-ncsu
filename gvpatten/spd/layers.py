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

#########################################################################

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
            if self.residual:
                x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
            else:
                x = self.norm[0](dh)
            dh_ffn = self.ff_func(x)
            if self.residual:
                x = self.norm[1](tuple_sum(x, self.dropout[1](dh_ffn)))
            else:
                x = self.norm[1](dh_ffn)
        
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
class GVPAttentionLayer(nn.Module):
    def __init__(
            self, 
            node_dims, 
            edge_h_dims,
            edge_in_dims,
            num_heads=4,
            max_spd=32,     # Graphormer RPE 的最大距离
            n_message=3,    # (用于 local_geom_conv)
            n_feedforward=2,# (用于 FFN 隐藏层维度)
            drop_rate=.1,
            max_path_len=10,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True, 
            norm_first=True                     
        ):
        
        super(GVPAttentionLayer, self).__init__()
        
        self.s_dim, self.v_dim = node_dims
        self.edge_s_in_dim, _ = edge_in_dims
        self.norm_first = norm_first
        self.num_heads = num_heads
        self.max_spd = max_spd
        self.local_geom_conv = GVPAttention( 
            node_dims, node_dims, edge_h_dims, n_message,
            aggr="mean", activations=activations, vector_gate=vector_gate
        )
        
        self.global_scalar_attn = nn.MultiheadAttention(
            embed_dim=self.s_dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True # (Confs, Nodes, Dim)
        )
        self.dropout_mha = Dropout(drop_rate) # (GVP-aware Dropout)
        
        self.spatial_bias = nn.Embedding(max_spd + 2, num_heads) # +2 for 0 and -1
        self.edge_encoder = EdgeEncoder(
            num_heads=num_heads, 
            edge_dim=self.edge_s_in_dim,
            max_path_len=max_path_len
        )
        self.fusion_gate_linear = nn.Linear(self.s_dim * 2, self.s_dim)
       
        self.norm_ffn = LayerNorm(node_dims) # (GVP-aware Norm)
        ffn_hid_dim = node_dims[0] * n_feedforward 
        self.ffn = PositionwiseFeedForward(
            d_in=self.s_dim, 
            d_hid=ffn_hid_dim, 
            dropout=drop_rate
        )
        self.dropout_ffn = Dropout(drop_rate) # (GVP-aware Dropout)
        self.norm1 = LayerNorm(node_dims) # (GVP-aware Norm)


    def forward(self, x, edge_index, edge_attr, shortest_path_edges, edge_features_s_all,
                spd_matrix, batch_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
                  s: [Nodes, Confs, s_dim]
                  v: [Nodes, Confs, v_dim, 3]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (edge_s, edge_v) of `torch.Tensor`
        :param spd_matrix: [Nodes, Nodes] 
        :param batch_mask: [Confs * Heads, Nodes, Nodes] (可选的, 0.0 或 -inf)
        '''
        x_res1 = x
        
        # --- Pre-Normalization ---
        x_norm = self.norm1(x) if self.norm_first else x
        s_norm, v_norm = x_norm

        # --- Local Path ---
        dh_local = self.local_geom_conv(x_norm, edge_index, edge_attr)
        s_local, v_local_update = dh_local 
        # s_local: [N, C, s_dim], v_local_update: [N, C, v_dim, 3]

        # --- Global Path ---
        s_in_global = s_norm.permute(1, 0, 2) # [Confs, Nodes, s_dim]
        n_confs, n_nodes, _ = s_in_global.shape

        spd_matrix_clamped = torch.clamp(spd_matrix + 1, min=0, max=self.max_spd + 1)
        spd_bias = self.spatial_bias(spd_matrix_clamped) # [N, N, H]
        spd_bias = spd_bias.permute(2, 0, 1)
        edge_bias = self.edge_encoder(edge_features_s_all, shortest_path_edges) # [H, N, N]
        total_bias = spd_bias + edge_bias        # [H, N, N]
        total_bias = total_bias.unsqueeze(0).repeat(n_confs, 1, 1, 1)
        total_bias = total_bias.view(n_confs * self.num_heads, n_nodes, n_nodes)

        if batch_mask is not None:
            total_bias = total_bias + batch_mask
        s_global_update, _ = self.global_scalar_attn(
            s_in_global, s_in_global, s_in_global,
            attn_mask=total_bias, 
            need_weights=False
        )
        # s_global_update: [C, N, s_dim] -> [N, C, s_dim]
        s_global_update = s_global_update.permute(1, 0, 2)

        gate_input = torch.cat([s_local, s_global_update], dim=-1)
        g = torch.sigmoid(self.fusion_gate_linear(gate_input))
        
        s_fused = g * s_local + (1 - g) * s_global_update

        x_update = (s_fused, v_local_update) 
        x_out = tuple_sum(x_res1, self.dropout_mha(x_update)) 
        
        if not self.norm_first: 
            x_out = self.norm1(x_out)

        x_res2 = x_out
        x_norm_ffn = self.norm_ffn(x_out) if self.norm_first else x_out
        s_ffn_in, v_ffn_in = x_norm_ffn

        s_ffn_out = self.ffn(s_ffn_in)

        s_update_ffn = self.dropout_ffn.sdropout(s_ffn_out) 
        v_update_ffn = torch.zeros_like(v_ffn_in) 
        x_update_ffn = (s_update_ffn, v_update_ffn)
        x_final = tuple_sum(x_res2, x_update_ffn)

        # --- FFN Post-Normalization ---
        if not self.norm_first:
            x_final = self.norm_ffn(x_final)
        
        return x_final



class GVPAttention(MessagePassing):
    '''
    GVPConv for handling multiple conformations
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVPAttention, self).__init__(aggr=aggr)
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
class EdgeEncoder(nn.Module):
    '''
    计算 Graphormer 边编码 (EE) 偏置 c_ij。
    '''
    def __init__(self, num_heads, edge_dim, max_path_len):
        super().__init__()
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.max_path_len = max_path_len
        self.edge_weights = nn.Embedding(max_path_len + 1, num_heads * edge_dim)

    def forward(self, edge_features_s, shortest_path_edges):
        N = shortest_path_edges.shape[0]
        device = edge_features_s.device

        valid_edge_indices = shortest_path_edges.clamp(min=0)
        # path_edge_features: [N, N, max_path_len, edge_s_dim]
        path_edge_features = edge_features_s[valid_edge_indices]

        path_positions = torch.arange(1, self.max_path_len + 1, device=device).unsqueeze(0).unsqueeze(0).expand(N, N, -1)
        # path_weights: [N, N, max_path_len, num_heads * edge_dim]
        path_weights = self.edge_weights(path_positions)
        # path_weights reshaped: [N, N, max_path_len, num_heads, edge_dim]
        path_weights = path_weights.view(N, N, self.max_path_len, self.num_heads, self.edge_dim)
        # path_edge_features expanded: [N, N, max_path_len, 1, edge_s_dim]
        path_edge_features = path_edge_features.unsqueeze(-2)
        # dot_products: [N, N, max_path_len, num_heads]
        dot_products = (path_edge_features * path_weights).sum(dim=-1)
        # path_mask: [N, N, max_path_len] (True for valid edges)
        path_mask = (shortest_path_edges != -1) 
        dot_products = dot_products * path_mask.unsqueeze(-1) 
        N_ij = path_mask.sum(dim=-1).clamp(min=1) 

        # sum_dot_products: [N, N, num_heads]
        sum_dot_products = dot_products.sum(dim=-2)
        
        # c_ij_per_head: [N, N, num_heads]
        c_ij_per_head = sum_dot_products / N_ij.unsqueeze(-1)
        edge_bias = c_ij_per_head.permute(2, 0, 1) # -> [num_heads, N, N]
        return edge_bias

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
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + 1e-8)
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