#####models.py changes######
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric

# Import layers
from src.layers import *
from src.mgnn.gps_layer import GPSLayer
from src.mgnn.adapters import GVP_to_GPS_Adapter, GPS_to_GVP_Adapter

# [Core modification] Create a custom EncoderBlock class to replace nn.Sequential
class EncoderBlock(nn.Module):
    def __init__(self, node_h_dim, edge_h_dim, drop_rate, num_heads, local_gnn_type, global_model_type):
        super().__init__()
        
        # Calculate various dimensions
        gvp_s_dim, gvp_v_dim = node_h_dim
        gps_node_dim = gvp_s_dim + gvp_v_dim * 3

        gvp_edge_s_dim, gvp_edge_v_dim = edge_h_dim
        gps_edge_dim = gvp_edge_s_dim + gvp_edge_v_dim * 3

        # Create three components internally
        self.adapter_in = GVP_to_GPS_Adapter(gvp_node_dims=node_h_dim, gvp_edge_dims=edge_h_dim)
        
        self.gps_layer = GPSLayer(
            dim_h=gps_node_dim,
            edge_dim_h=gps_edge_dim,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            num_heads=num_heads,
            batch_norm=True,
            dropout=drop_rate
        )

        self.adapter_out = GPS_to_GVP_Adapter(
            gvp_node_dims=node_h_dim, 
            gvp_edge_dims=edge_h_dim,
            gps_node_dim=gps_node_dim,
            gps_edge_dim=gps_node_dim
        )
    
    def forward(self, h_V, h_E, batch):
        # Explicitly define data flow
        gps_batch = self.adapter_in(h_V, h_E, batch)
        gps_out_batch = self.gps_layer(gps_batch)
        h_V_out, h_E_out = self.adapter_out(gps_out_batch)
        return h_V_out, h_E_out
