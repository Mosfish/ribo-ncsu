################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric

from src.layers import *


class AutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    
    MODIFIED: Uses GVPAttentionLayer (V5) in the encoder.
    '''
    def __init__(
            self,
            node_in_dim = (64, 4), 
            node_h_dim = (128, 16), 
            edge_in_dim = (32, 1), 
            edge_h_dim = (32, 1),
            num_layers = 3, 
            drop_rate = 0.1,
            out_dim = 4,
            num_heads = 4,      
            max_spd = 32,      
            max_path_len = 10,  

    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim_encoder = edge_h_dim 
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None) 
        
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim_encoder, 
                activations=(None, None), vector_gate=True)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPAttentionLayer( 
                    node_dims=node_h_dim, 
                    edge_h_dims=self.edge_h_dim_encoder,
                    edge_in_dims=self.edge_in_dim,
                    num_heads=num_heads,        
                    max_spd=max_spd,        
                    max_path_len=max_path_len,  
                    n_message=3,            
                    n_feedforward=2,        
                    drop_rate=drop_rate,
                    norm_first=True    # (GVPAttentionLayer default True)
                )
            for _ in range(num_layers))

        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        self.edge_h_dim_decoder = (self.edge_h_dim_encoder[0] + self.out_dim, self.edge_h_dim_encoder[1]) 
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_h_dim, 
                    self.edge_h_dim_decoder,
                    activations=(F.silu, None), 
                    vector_gate=True, 
                    drop_rate=drop_rate, 
                    autoregressive=True, 
                    norm_first=True
                ) 
            for _ in range(num_layers))

        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):
        '''
        训练和评估似然时使用。
        '''
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq
        try:
            spd_matrix = batch.spd_matrix            # [N, N]
            shortest_path_edges = batch.shortest_path_edges # [N, N, max_path_len]

            mask_confs = batch.mask_confs
            # --- 修复开始：健壮性检查 ---
            if mask_confs is None or mask_confs.size(1) == 1:
                if batch.edge_s.dim() == 3:
                    edge_features_s_all = batch.edge_s[:, 0] # [E, C, dim] -> [E, dim]
                else:
                    edge_features_s_all = batch.edge_s       # 已经是 [E, dim]
            else:
            # --- 修复结束 ---
                n_conf_true = mask_confs.sum(1, keepdim=True).clamp(min=1) 
                n_conf_true_edges = n_conf_true[edge_index[0]] 
                mask_nodes_s = mask_confs.unsqueeze(2) 
                mask_edges_s = mask_nodes_s[edge_index[0]] 
                edge_features_s_all = (batch.edge_s * mask_edges_s).sum(dim=1) / n_conf_true_edges

        except AttributeError as e:
            raise AttributeError(f"need {e} from batch.")
        
        # 必须在 W_v 之前获取 num_confs
        num_confs = h_V[0].shape[1]
        
        num_nodes = batch.num_nodes
        num_heads = self.encoder_layers[0].num_heads
        device = edge_index.device
        
        batch_vec = batch.batch.to(device)
        batch_mask_bool = (batch_vec.unsqueeze(1) == batch_vec.unsqueeze(0)) # [N, N]
        batch_mask = torch.full(
            (num_nodes, num_nodes), 
            -torch.inf, 
            device=device, 
            dtype=torch.float32
        )
        batch_mask[batch_mask_bool] = 0.0
        batch_mask = batch_mask.unsqueeze(0).repeat(num_confs * num_heads, 1, 1)
        
        h_V = self.W_v(h_V)
        h_E_encoder = self.W_e(h_E) 
        for layer in self.encoder_layers:
            h_V = layer(
                h_V, 
                edge_index, 
                h_E_encoder, 
                shortest_path_edges=shortest_path_edges, 
                edge_features_s_all=edge_features_s_all, 
                spd_matrix=spd_matrix,       
                batch_mask=batch_mask 
            ) 
        h_V_pooled, h_E_pooled = self.pool_multi_conf(h_V, h_E_encoder, batch.mask_confs, edge_index)

        encoder_embeddings = h_V_pooled
        
        h_S = self.W_s(seq) 
        h_S = h_S[edge_index[0]] 
        # Masking for autoregressive property
        h_S[edge_index[0] >= edge_index[1]] = 0 

        h_E_decoder = (torch.cat([h_E_pooled[0], h_S], dim=-1), h_E_pooled[1]) 
        
        h_V_decoder_in = h_V_pooled 
        for layer in self.decoder_layers:
            h_V_decoder_in = layer(
                h_V_decoder_in,
                edge_index,
                h_E_decoder,
                autoregressive_x=encoder_embeddings
            )
        logits = self.W_out(h_V_decoder_in) 
        
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
        ):
        '''
        Samples sequences autoregressively.
        MODIFIED: Needs to pass V5 inputs to the encoder loop.
        '''
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        try:
            spd_matrix = batch.spd_matrix        
            shortest_path_edges = batch.shortest_path_edges 

            mask_confs = batch.mask_confs
            # --- 修复开始：健壮性检查 ---
            if mask_confs is None or mask_confs.size(1) == 1:
                if batch.edge_s.dim() == 3:
                    edge_features_s_all = batch.edge_s[:, 0] # [E, C, dim] -> [E, dim]
                else:
                    edge_features_s_all = batch.edge_s       # 已经是 [E, dim]
            else:
            # --- 修复结束 ---
                n_conf_true = mask_confs.sum(1, keepdim=True).clamp(min=1) 
                n_conf_true_edges = n_conf_true[edge_index[0]] 
                mask_nodes_s = mask_confs.unsqueeze(2) 
                mask_edges_s = mask_nodes_s[edge_index[0]] 
                edge_features_s_all = (batch.edge_s * mask_edges_s).sum(dim=1) / n_conf_true_edges

        except AttributeError as e:
            raise AttributeError(f"model need {e} from batch.")

        device = edge_index.device
        
        # 必须在 W_v 之前获取 num_confs
        num_confs = h_V[0].shape[1]
        
        num_nodes = h_V[0].shape[0]
        num_heads = self.encoder_layers[0].num_heads
        
        batch_vec = batch.batch.to(device)
        batch_mask_bool = (batch_vec.unsqueeze(1) == batch_vec.unsqueeze(0)) # [N, N]
        batch_mask = torch.full(
            (num_nodes, num_nodes), 
            -torch.inf, 
            device=device, 
            dtype=torch.float32
        )
        batch_mask[batch_mask_bool] = 0.0
        batch_mask = batch_mask.unsqueeze(0).repeat(num_confs * num_heads, 1, 1)
        h_V = self.W_v(h_V) 
        h_E_encoder = self.W_e(h_E) 
        
        for layer in self.encoder_layers:
            h_V = layer(
                h_V, 
                edge_index, 
                h_E_encoder, 
                shortest_path_edges=shortest_path_edges,  
                edge_features_s_all=edge_features_s_all,  
                spd_matrix=spd_matrix,              
                batch_mask=batch_mask           
            ) 

        h_V_pooled, h_E_pooled = self.pool_multi_conf(h_V, h_E_encoder, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V_pooled = (h_V_pooled[0].repeat(n_samples, 1),
                h_V_pooled[1].repeat(n_samples, 1, 1))
        h_E_pooled = (h_E_pooled[0].repeat(n_samples, 1),
                h_E_pooled[1].repeat(n_samples, 1, 1))
        
        # Expand edge index
        edge_index_expanded = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index_expanded = torch.cat(tuple(edge_index_expanded + offset), dim=-1)
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        # Cache encoder embeddings for decoder
        encoder_embeddings_repeated = (h_V_pooled[0].clone(), h_V_pooled[1].clone())
        h_V_cache = [(encoder_embeddings_repeated[0].clone(), encoder_embeddings_repeated[1].clone()) 
                     for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            
            # Prepare edge features with previous predictions
            h_S_ = h_S[edge_index_expanded[0]] 
            h_S_[edge_index_expanded[0] >= edge_index_expanded[1]] = 0
            h_E_decoder = (torch.cat([h_E_pooled[0], h_S_], dim=-1), h_E_pooled[1])
                      
            # Masking for current node i
            edge_mask = edge_index_expanded[1] % num_nodes == i 
            edge_index_ = edge_index_expanded[:, edge_mask] 
            h_E_ = tuple_index(h_E_decoder, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True 
            
            out = None
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                            autoregressive_x=encoder_embeddings_repeated, # Use repeated encoder embeddings
                            node_mask=node_mask)
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][node_mask] = out[0][node_mask] # Update only node i's scalar
                    h_V_cache[j+1][1][node_mask] = out[1][node_mask] # Update only node i's vector
            
            lgts = self.W_out(tuple_index(out, node_mask)) # Get output only for node i
            if logit_bias is not None:
                lgts += logit_bias[i]
            
            current_samples = Categorical(logits=lgts / temperature).sample()
            seq[i::num_nodes] = current_samples
            h_S[i::num_nodes] = self.W_s(current_samples)
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:   
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        """
        Pools features across multiple conformations using masked average pooling.
        """
        if mask_confs is None or mask_confs.size(1) == 1:
            # Handle case with no mask or only one conformation
            if h_V[0].dim() == 3: # Check if pooling is needed
                return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
            else:
                return h_V, h_E # Already pooled
        
        n_conf_true = mask_confs.sum(1, keepdim=True).clamp(min=1) # (N, 1) Prevent div by zero
        
        mask_nodes_s = mask_confs.unsqueeze(2) # (N, C, 1)
        mask_nodes_v = mask_nodes_s.unsqueeze(3) # (N, C, 1, 1)
        mask_edges_s = mask_nodes_s[edge_index[0]] # (E, C, 1)
        mask_edges_v = mask_nodes_v[edge_index[0]] # (E, C, 1, 1)
        
        h_V0 = (h_V[0] * mask_nodes_s).sum(dim=1) / n_conf_true
        h_V1 = (h_V[1] * mask_nodes_v).sum(dim=1) / n_conf_true.unsqueeze(2)
        
        n_conf_true_edges = n_conf_true[edge_index[0]] # (E, 1)
        h_E0 = (h_E[0] * mask_edges_s).sum(dim=1) / n_conf_true_edges
        h_E1 = (h_E[1] * mask_edges_v).sum(dim=1) / n_conf_true_edges.unsqueeze(2)

        return (h_V0, h_V1), (h_E0, h_E1)


class NonAutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Non-Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        node_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Output
        self.W_out = torch.nn.Sequential(
            LayerNorm(self.node_h_dim),
            GVP(self.node_h_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True),
            GVP(self.node_h_dim, (self.out_dim, 0), 
                activations=(None, None))   
        )
    
    def forward(self, batch):

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        # h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))

        logits = self.W_out(h_V)  # (n_nodes, out_dim)
        
        return logits
    
    def sample(self, batch, n_samples, temperature=0.1, return_logits=False):
        
        with torch.no_grad():

            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            edge_index = batch.edge_index
        
            h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
            h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
            
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
            
            # Pool multi-conformation features
            # h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
            h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
            
            logits = self.W_out(h_V)  # (n_nodes, out_dim)
            probs = F.softmax(logits / temperature, dim=-1)
            seq = torch.multinomial(probs, n_samples, replacement=True)  # (n_nodes, n_samples)

            if return_logits:
                return seq.permute(1, 0).contiguous(), logits.unsqueeze(0).repeat(n_samples, 1, 1)
            else:
                return seq.permute(1, 0).contiguous()
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):

        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)
        
        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        # Average pooling multi-conformation features
        h_V = (h_V0.sum(dim=1) / n_conf_true,               # (n_nodes, d_s)
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))  # (n_nodes, d_v, 3)
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],               # (n_edges, d_se)
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))  # (n_edges, d_ve, 3)

        return h_V, h_E