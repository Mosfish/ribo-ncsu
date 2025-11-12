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
        heads = 4,
        out_dim = 4,
        n_anchors = 32,
        max_degree = 100,      # 我们关心的最大度数
        degree_emb_dim = 16   # 用于编码度数的嵌入维度
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.heads = heads
        self.out_dim = out_dim
        activations = (F.silu, None)

        # --- [V7.3 新增] ---
        # 定义度数编码器 (Graphormer 中心性编码)
        self.degree_encoder = nn.Embedding(max_degree + 1, degree_emb_dim)
        self.max_degree = max_degree
        # --- [V7.3 新增结束] ---
        # 添加标准初始化
        nn.init.normal_(self.degree_encoder.weight, std=0.02)

        self.degree_norm = LayerNorm((degree_emb_dim, 0))
        # --- [V7.3 修改] ---
        # 计算增广后的节点输入维度
        # s_dim = 64 (原始) + 16 (度数)
        augmented_node_s_dim = self.node_in_dim[0] + degree_emb_dim
        augmented_node_in_dim = (augmented_node_s_dim, self.node_in_dim[1])

        # Node input embedding (W_v)
        # 必须修改 W_v 来接收增广后的维度
        self.W_v = torch.nn.Sequential(
            LayerNorm(augmented_node_in_dim), # <--- 修改
            GVP(augmented_node_in_dim, self.node_h_dim, # <--- 修改
                activations=(None, None), vector_gate=True)
        )
        # --- [V7.3 修改结束] ---

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                GeoLSTLayer(self.node_h_dim, self.edge_h_dim, 
                                  heads=self.heads,
                                  n_anchors=n_anchors, # 传入新参数
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        
        # [V6] 良好实践：隔离解码器的边维度，防止 self.edge_h_dim 被污染
        decoder_edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, decoder_edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output (unchanged)
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):

        h_V_orig = (batch.node_s, batch.node_v) # [V7.3 修改] 重命名
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq
        
        # --- [V7.3 新增：中心性编码注入] ---
        s_orig, v_orig = h_V_orig
        
        # 1. 获取度数, 形状 [N] (假设 featurizer.py 已添加)
        deg_indices = batch.degree
        
        # 2. 安全地 clamp 到最大值
        deg_indices = deg_indices.clamp(max=self.max_degree)
        
        # 3. 编码度数: [N] -> [N, degree_emb_dim]
        deg_emb = self.degree_encoder(deg_indices)
        deg_emb = self.degree_norm(deg_emb)
        # 4. 扩展以匹配 n_conf 维度
        # s_orig 形状: [N, n_conf, s_dim]
        n_conf = s_orig.shape[1]
        deg_emb_expanded = deg_emb.unsqueeze(1).expand(-1, n_conf, -1)
        
        # 5. 拼接标量特征 (在最后一个维度)
        s_augmented = torch.cat([s_orig, deg_emb_expanded], dim=-1)
        
        # 6. 创建增广 GVP 元组
        h_V_augmented = (s_augmented, v_orig)
        # --- [V7.3 注入结束] ---

        # [V7.3 修改] h_V 现在由增广后的元组生成
        h_V = self.W_v(h_V_augmented)
        h_E = self.W_e(h_E)

        # (从这里开始, h_V 已携带拓扑信息, 无需再改)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E) 

        # Pool multi-conformation features: 
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        
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
        [V7.3] 此函数需要与 forward 保持一致
        '''
        h_V_orig = (batch.node_s, batch.node_v) # [V7.3 修改]
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V_orig[0].shape[0] # [V7.3 修改]
        
        # --- [V7.3 新增：中心性编码注入] ---
        s_orig, v_orig = h_V_orig
        deg_indices = batch.degree
        deg_indices = deg_indices.clamp(max=self.max_degree)
        deg_emb = self.degree_encoder(deg_indices)
        deg_emb = self.degree_norm(deg_emb)
        n_conf = s_orig.shape[1]
        deg_emb_expanded = deg_emb.unsqueeze(1).expand(-1, n_conf, -1)
        s_augmented = torch.cat([s_orig, deg_emb_expanded], dim=-1)
        h_V_augmented = (s_augmented, v_orig)
        # --- [V7.3 注入结束] ---

        # [V7.3 修改]
        h_V = self.W_v(h_V_augmented)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Pool multi-conformation features
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))
        
        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                    
            edge_mask = edge_index[1] % num_nodes == i
            edge_index_ = edge_index[:, edge_mask]
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True
            
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                
                out = tuple_index(out, node_mask)
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
                
            lgts = self.W_out(out)
            if logit_bias is not None:
                lgts += logit_bias[i]
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
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
