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
class GVPAttentionConvLayer(nn.Module):
    '''
    Full GVP Attention Layer
    
    Applies 'GVPAttentionConv' message passing, followed by
    residual updates and a pointwise feedforward network.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param heads: number of attention heads
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
    :param residual: whether to use residual connections
    :param norm_first: whether to apply layer norm before 'conv' and 'ff'
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
            activations=activations, 
            vector_gate=vector_gate
        )
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # Standard GVP FeedForward network (identical to MultiGVPConvLayer)
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
#########################################################################
class GVPAttentionConv(MessagePassing):
    '''
    GVP-Native Attention-based Message Passing
    - Q, K, V projections are GVP-native.
    - Attention energy is computed by a GVP-based 'head'.
    - Edge features (GVP tuples) are used to bias K and V.
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 heads=4,
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

        # Attention Head
        # Input: concatenated Q_i and K_j_biased (per-head)
        attn_in_dims = (2 * self.head_s_dim, 2 * self.head_v_dim)
        # Output: 1 scalar (energy) per head
        self.attn_gvp = GVP(attn_in_dims, (1, 0), 
                            activations=(None, None))

        # Output Projection (Multi-head fusion)
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

        # Project Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Project Edge Biases
        e_k = self.edge_proj_k(edge_attr)
        e_v = self.edge_proj_v(edge_attr)
        
        # --- Flatten all tuples for MessagePassing ---
        # We follow the GVPConv pattern of flattening vectors for propagation
        # (s, v) -> (s, v_flat) -> cat(s, v_flat)
        
        def flatten_heads(t, n, n_conf):
            s, v = t
            # .reshape() is robust and handles both contiguous and non-contiguous cases.
            s_flat = s.reshape(n, n_conf * self.so)
            v_flat = v.reshape(n, n_conf * self.vo * 3)
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
        
        # Helper to un-flatten and reshape to heads
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

        # Compute Attention Energy
        # attn_input: (s_cat, v_cat)
        attn_input = tuple_cat(q_i, k_j_biased)
        
        # self.attn_gvp maps (2*head_dims) -> (heads, 0)
        # energy_s shape: [E, n_conf, H, 1]
        energy_s = self.attn_gvp(attn_input)
        energy_scalar = energy_s.squeeze(-1) # [E, n_conf, H]
        # We need to softmax over neighbors j for each node i.
        # Reshape for softmax: [E, n_conf * H]
        energy_flat = energy_scalar.view(-1, n_conf * self.heads)
        alpha_flat = softmax(energy_flat, index, ptr=ptr, dim=0)
        # Reshape back: [E, n_conf, H]
        alpha = alpha_flat.view(-1, n_conf, self.heads)

        # Apply Attention to Biased Value
        # alpha_s: [E, n_conf, H, 1]
        alpha_s = alpha.unsqueeze(-1)
        # alpha_v: [E, n_conf, H, 1, 1]
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        v_s_biased, v_v_biased = v_j_biased
        msg_s = alpha_s * v_s_biased # [E, n_conf, H, s_h]
        msg_v = alpha_v * v_v_biased # [E, n_conf, H, v_h, 3]

        # --- Flatten message for output ---
        # msg_s: [E, n_conf * H * s_h] = [E, n_conf * so]
        msg_s_flat = msg_s.contiguous().view(msg_s.shape[0], -1)
        # msg_v: [E, n_conf * H * v_h * 3] = [E, n_conf * vo * 3]
        msg_v_flat = msg_v.contiguous().view(msg_v.shape[0], -1)

        return torch.cat([msg_s_flat, msg_v_flat], dim=-1)

########################################################################
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
