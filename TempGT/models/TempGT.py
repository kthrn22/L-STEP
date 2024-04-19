import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.sparse as sparse
import torch.fft as fft

from linformer import Linformer
from models.modules import TimeEncoder

# class TimeEncoder(nn.Module):
#     def __init__(self, time_dim, alpha, beta):
#         super(TimeEncoder, self).__init__()
#         self.time_vec = alpha ** ((-torch.arange(1, time_dim + 1) + 1) / beta)

#     def forward(self, timestamps):
#         t_e = timestamps * self.time_vec
#         t_e = torch.cos(t_e)

#         return t_e

class TempGT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler, pe_dim: int,
                 num_neighbors: int, time_feat_dim: int, num_batches: int, seq_len: int, num_heads: int, 
                 transformer_depth: int, num_edge_layers: int = 2 , dropout: float = 0.1, 
                 transformer: str = "linformer", device: str = 'cuda'):
        
        super(TempGT, self).__init__()

        edge_feat_dim = edge_raw_features.shape[-1]
        node_feat_dim = node_raw_features.shape[-1]
        self.num_batches = num_batches                      # T
        # self.positional_encoding = 
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = TimeEncoder(time_feat_dim, parameter_requires_grad = False)
        self.device = device

        self.fft_filter = nn.Linear(pe_dim, pe_dim, bias = False)
        self.fft_dropout = nn.Dropout(p = dropout)
        self.fft_agg = nn.Linear(num_batches, 1, bias = False)

        self.edge_mlp_layers = nn.ModuleList(
            nn.Linear(time_feat_dim + edge_feat_dim, time_feat_dim + edge_feat_dim) for _ in range(num_edge_layers)
        )
        self.edge_mlp = nn.Linear(edge_feat_dim + time_feat_dim, edge_feat_dim)
        self.edge_final_mlp = nn.Linear(num_neighbors, 1)

        self.node_final_mlp = nn.Linear(edge_feat_dim + node_feat_dim, node_feat_dim)

        self.transformer = Linformer(dim = node_feat_dim, seq_len = seq_len, 
                                depth = transformer_depth, heads = num_heads, k = 256, one_kv_head = True, share_kv = True)

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def fourier_transform_pe(self, node_ids, pe, batch_idx, use_dropout = True, use_mixer = False):
        batch_pe = pe[torch.from_numpy(node_ids)]       # [batch_size x batch_idx x pe_dim]
        padding = torch.zeros((batch_pe.shape[0], self.num_batches - batch_pe.shape[1], batch_pe.shape[-1])).to(self.device)
        batch_pe = torch.cat([batch_pe, padding], dim = 1) # [batch_size x T x pe_dim]
        
        mask = torch.zeros(batch_pe.shape).to(self.device)
        mask[:, : batch_idx, :] += 1                                  
        
        init_pe = torch.clone(batch_pe)
        batch_pe = batch_pe.to(torch.complex64)
        batch_pe = fft.fftn(batch_pe, dim = 1)
        batch_pe *= mask
        
        batch_pe = batch_pe.to(torch.float32)

        batch_pe = self.fft_filter(batch_pe)
        batch_pe *= mask
        batch_pe = fft.ifftn(batch_pe, dim = 1)               # [N x T x pe_dim]
        batch_pe *= mask
        batch_pe = batch_pe.to(torch.float32)
        
        if use_dropout:
            batch_pe = self.fft_dropout(batch_pe)
            batch_pe += init_pe

        if use_mixer:
            return                                                      # implementation needed

        current_pe = self.fft_agg(batch_pe.permute(0, 2, 1)).squeeze()     # [N x T x pe_dim] -> [N x pe_dim x T] -> [N x pe_dim]
       
        return current_pe                                                       # [N x pe_dim]
    
    def aggregated_node_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
        num_neighbors: int = 20, time_gap: int = 2000):
        '''
        obtain node & edge embeddings
        return aggregated node embeddings for each node
        '''
        # get links
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
                                                           node_interact_times = node_interact_times,
                                                           num_neighbors = num_neighbors)

        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # (N, num_neighbors, edge_feat_dim + time_feat_dim) -> (N, num_neighbors, edge_feat_dim + time_feat_dim)
        
        # for mlp in self.edge_mlp_layers:
        #     combined_features = mlp(combined_features)
        ### simple MLPs or MLP - Mixer?, or single MLP ?
        combined_features = self.edge_mlp(combined_features)    # (N, edge_dim + time_dim -> edge_dim), single MLP layer
    
        # (N, num_neighbors, edge_feat_dim) -> (N, edge_feat_dim, num_neighbors) 
        # -> (N, edge_feat_dim, 1) -> (N, edge_feat_dim)
        combined_features = self.edge_final_mlp(combined_features.permute(0, 2, 1)).squeeze()
        # return combined_features
    
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                          node_interact_times=node_interact_times,
                                                                                          num_neighbors=time_gap)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(time_gap_neighbor_node_ids)]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.node_final_mlp(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings
    
    def linear_transformer(self, pe, node_ids: np.ndarray, node_embeddings: torch.tensor):
        
        
        if pe is not None:
            node_embeddings += pe[torch.from_numpy(node_ids)]                         # (batch_size, node_feat_dim)

        return self.transformer(node_embeddings.unsqueeze(0)).squeeze(0)             # (1, batch_size, node_feat_dim)    
    
    def compute_node_embeddings(self, pe, node_ids, node_interact_times, num_neighbors, time_gap):
        node_embeddings = self.aggregated_node_embeddings(node_ids, node_interact_times, num_neighbors, time_gap)
        node_embeddings = self.linear_transformer(pe, node_ids, node_embeddings)

        return node_embeddings