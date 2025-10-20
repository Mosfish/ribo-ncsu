# File path: src/layers/adapters.py

import torch
import torch.nn as nn
from torch_geometric.data import Batch

class GVP_to_GPS_Adapter(nn.Module):
    def __init__(self, gvp_node_dims, gvp_edge_dims):
        super().__init__()
        self.node_s_dim, self.node_v_dim = gvp_node_dims
        self.edge_s_dim, self.edge_v_dim = gvp_edge_dims

    def forward(self, gvp_node_data, gvp_edge_data, batch_obj):
        # --- Process node features ---
        s_node, v_node = gvp_node_data
        
        # [Key modification] Flatten both scalar and vector parts to 2D
        s_node_flat = s_node.contiguous().view(s_node.shape[0], -1)
        v_node_flat = v_node.contiguous().view(v_node.shape[0], -1)
        
        # Now both are 2D tensors, can safely concatenate
        x = torch.cat([s_node_flat, v_node_flat], dim=-1)
        
        # --- Apply same processing to edge features for robustness ---
        s_edge, v_edge = gvp_edge_data
        
        s_edge_flat = s_edge.contiguous().view(s_edge.shape[0], -1)
        v_edge_flat = v_edge.contiguous().view(v_edge.shape[0], -1)

        edge_attr = torch.cat([s_edge_flat, v_edge_flat], dim=-1)

        # --- Create new Batch object (logic unchanged) ---
        new_batch = batch_obj.clone()
        new_batch.x = x
        new_batch.edge_attr = edge_attr

        if not hasattr(new_batch, 'batch') or new_batch.batch is None:
            # If batch attribute doesn't exist or is None, create a zero tensor
            # This means all nodes in the batch belong to the same graph (graph 0)
            num_nodes = new_batch.num_nodes if hasattr(new_batch, 'num_nodes') else new_batch.x.size(0)
            new_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=new_batch.x.device)

        return new_batch

class GPS_to_GVP_Adapter(nn.Module):
    def __init__(self, gvp_node_dims, gvp_edge_dims, gps_node_dim, gps_edge_dim):
        super().__init__()
        self.node_s_dim, self.node_v_dim = gvp_node_dims
        self.edge_s_dim, self.edge_v_dim = gvp_edge_dims
        
        # [Core modification] Create linear projection layers for dimensionality reduction
        # Node feature projection layer
        gvp_node_total_dim = self.node_s_dim + self.node_v_dim * 3
        self.node_proj = nn.Linear(gps_node_dim, gvp_node_total_dim)

        # Edge feature projection layer
        gvp_edge_total_dim = self.edge_s_dim + self.edge_v_dim * 3
        self.edge_proj = nn.Linear(gps_edge_dim, gvp_edge_total_dim)

    def forward(self, gps_out_batch):
        # --- Process node features ---
        x = gps_out_batch.x # dimension: [n_nodes, 176]
        x_proj = self.node_proj(x) # project to: [n_nodes, 176] (assuming node dimension unchanged)
        
        s_node = x_proj[:, :self.node_s_dim]
        v_node_flat = x_proj[:, self.node_s_dim:]
        v_node = v_node_flat.contiguous().view(v_node_flat.shape[0], self.node_v_dim, 3)
        
        # --- Process edge features ---
        edge_attr = gps_out_batch.edge_attr # dimension: [n_edges, 176]
        edge_attr_proj = self.edge_proj(edge_attr) # [Core modification] project to: [n_edges, 76]
        
        s_edge = edge_attr_proj[:, :self.edge_s_dim]
        v_edge_flat = edge_attr_proj[:, self.edge_s_dim:]
        v_edge = v_edge_flat.contiguous().view(v_edge_flat.shape[0], self.edge_v_dim, 3)
        
        return (s_node, v_node), (s_edge, v_edge)
