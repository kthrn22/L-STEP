import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.sparse as sparse
import torch.fft as fft
import pdb
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
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler, full_neighbor_sampler, pe_dim: int,
                 num_neighbors: int, time_feat_dim: int, num_fft_batches: int, seq_len: int, num_heads: int, 
                 transformer_depth: int, num_layers: int = 2 , dropout: float = 0.1, 
                 transformer: str = "linformer", device: str = 'cuda'):
        
        super(TempGT, self).__init__()

        edge_feat_dim = edge_raw_features.shape[-1]
        node_feat_dim = node_raw_features.shape[-1]
        self.num_fft_batches = num_fft_batches 
        self.num_nodes = node_raw_features.shape[0]
        # self.positional_encoding = 
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.full_neighbor_sampler = full_neighbor_sampler
        self.time_encoder = TimeEncoder(time_feat_dim, parameter_requires_grad = False)
        self.device = device

        self.fft_filter = nn.Linear(pe_dim, pe_dim, bias = False).to(torch.complex64)
        self.fft_dropout = nn.Dropout(p = dropout)
        self.fft_agg = nn.Linear(num_fft_batches, 1, bias = False)

        self.pe_mlp = nn.Linear(2 * pe_dim, pe_dim)
        self.pe_mlp_activation_func = nn.ReLU()

        self.edge_mlp_layers = nn.ModuleList(
            nn.Linear(time_feat_dim + edge_feat_dim, time_feat_dim + edge_feat_dim) for _ in range(num_layers)
        )
        self.edge_mlp = nn.Linear(edge_feat_dim + time_feat_dim, edge_feat_dim)
        
        self.edge_final_mlp = nn.Linear(num_neighbors, 1)

        self.node_final_mlp = nn.Linear(edge_feat_dim + node_feat_dim, node_feat_dim)

        self.transformer = Linformer(dim = node_feat_dim, seq_len = seq_len, 
                                depth = transformer_depth, heads = num_heads, k = 256, one_kv_head = True, share_kv = True)
        
        self.node_embeddings_mlps = nn.ModuleList(
            [nn.Linear(2 * node_feat_dim, node_feat_dim),
             nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim)]
            # nn.Linear(node_feat_dim, node_feat_dim) for _ in range(num_edge_layers)
            )

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

    def fourier_transform_pe(self, node_ids, pe, batch_idx, use_dropout = False, use_mixer = False):
        batch_pe = pe[torch.from_numpy(node_ids)]       # [batch_size x batch_idx x pe_dim]
        mask = None
        
        if batch_pe.shape[1] < self.num_fft_batches:
            padding = torch.zeros((batch_pe.shape[0], self.num_fft_batches - batch_pe.shape[1], batch_pe.shape[-1])).to(self.device)
            batch_pe = torch.cat([batch_pe, padding], dim = 1) # [batch_size x T x pe_dim]
        
            mask = torch.zeros(batch_pe.shape).to(self.device)
            mask[:, : batch_idx, :] += 1                                  
        
        init_pe = torch.clone(batch_pe)
        batch_pe = batch_pe.to(torch.complex64)
        batch_pe = fft.fftn(batch_pe, dim = 1)
        if mask is not None:
            batch_pe *= mask
        
        # batch_pe = batch_pe.to(torch.float32)

        batch_pe = self.fft_filter(batch_pe)
        if mask is not None:
            batch_pe *= mask
        
        batch_pe = fft.ifftn(batch_pe, dim = 1)               # [N x T x pe_dim]
        if mask is not None:
            batch_pe *= mask
        
        batch_pe = batch_pe.to(torch.float32)
        
        if use_dropout:
            batch_pe = self.fft_dropout(batch_pe)
            batch_pe += init_pe

        if use_mixer:
            return                                                      # implementation needed

        current_pe = self.fft_agg(batch_pe.permute(0, 2, 1)).squeeze()     # [N x num_fft_batches x pe_dim] -> [N x pe_dim x num_fft_batches] -> [N x pe_dim]
       
        return current_pe                                                       # [N x pe_dim]
    
    def aggregated_node_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
        num_neighbors: int = 20, time_gap: int = 2000, full = False):
        '''
        obtain node & edge embeddings
        return aggregated node embeddings for each node
        '''
        # get links
        if full is False:
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
                                                            node_interact_times = node_interact_times,
                                                            num_neighbors = num_neighbors)
        else:
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.full_neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
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
        
        if full == False:
            time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                          node_interact_times=node_interact_times,
                                                                                          num_neighbors=time_gap)
        else:
            time_gap_neighbor_node_ids, _, _ = self.full_neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
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
    
    def aggregated_pe(self, pe, node_ids: np.ndarray, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        # [batch_size, batch_size]
        # mapping unique node ids -> [0, 199]

        mapping_dict = {}
        for idx, val in enumerate(node_ids):
            mapping_dict[val] = idx
        
        mapped_src = np.array([mapping_dict[i] for i in batch_src_node_ids])
        mapped_dst = np.array([mapping_dict[i] for i in batch_dst_node_ids])

        batch_adj = torch.sparse_coo_tensor(indices = [mapped_src.tolist() + mapped_dst.tolist(),
                                                       mapped_dst.tolist() + mapped_src.tolist()],
                                            values = torch.ones(mapped_src.shape[0] + mapped_dst.shape[0]),
                                            ).to_dense().to(self.device)
        
        batch_pe = pe[node_ids]
        
        aggregated_pe = torch.mm(batch_adj, batch_pe) / batch_adj.sum(dim = -1).unsqueeze(-1)
        
        aggregated_pe = torch.concat([batch_pe, aggregated_pe], dim = -1)
        aggregated_pe = self.pe_mlp(aggregated_pe)
        out_pe = self.pe_mlp_activation_func(aggregated_pe)
    

        # if full == False:
        #     time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
        #                                                                                   node_interact_times=node_interact_times,
        #                                                                                   num_neighbors=time_gap)
        # else:
        #     time_gap_neighbor_node_ids, _, _ = self.full_neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
        #                                                                                   node_interact_times=node_interact_times,
        #                                                                                   num_neighbors=time_gap)
    
        # nodes_time_gap_neighbor_node_pe = pe[torch.from_numpy(time_gap_neighbor_node_ids)]
        # # Tensor, shape (batch_size, time_gap)
        # valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        # valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # # Tensor, shape (batch_size, time_gap)
        # scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)
        # # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        # nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_pe * scores.unsqueeze(dim=-1), dim=1)
        # # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        # output_node_pe = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]
        # # Tensor, shape (batch_size, node_feat_dim)
        # # node_embeddings = self.node_final_mlp(torch.cat([combined_features, output_node_features], dim=1))
        # aggregated_node_pe = self.pe_mlp(output_node_pe)

        return out_pe

    ### old version
    # def linear_transformer(self, pe, node_ids: np.ndarray, node_embeddings: torch.tensor):
    #     if pe is not None:
    #         node_embeddings += pe[torch.from_numpy(node_ids)]                         # (batch_size, node_feat_dim)

    #     return self.transformer(node_embeddings.unsqueeze(0)).squeeze(0)             # (1, batch_size, node_feat_dim)   

    ###* Applying Graph Transformer on temporal graphs with information accumulated from 1st batch to (i - 1)-batch 
    ### to obtain spatial representation for i-th batch
    def linear_transformer(self, pe, min_time, num_neighbors, time_gap, train_node_ids = None):
        # if pe is not None:
        #     node_embeddings += pe[torch.from_numpy(node_ids)]                         # (batch_size, node_feat_dim)

        # return self.transformer(node_embeddings.unsqueeze(0)).squeeze(0)             # (1, batch_size, node_feat_dim)  
        # (self, node_ids: np.ndarray, node_interact_times: np.ndarray,
        # num_neighbors: int = 20, time_gap: int = 2000)  
        node_ids = torch.arange(self.node_raw_features.shape[0]).to(torch.int64).numpy()
        node_interact_times = torch.Tensor([min_time]).broadcast_to(node_ids.shape).numpy()
        train_mask = None
        
        ### Tell transformer to mask out the new node (nodes not appeared in training set)
        ### avoiding info leakage
        if train_node_ids is not None:
            train_mask = torch.zeros(self.node_raw_features.shape)
            train_mask_ids = torch.from_numpy(train_node_ids).unsqueeze(-1).broadcast_to((train_node_ids.shape[0], self.node_raw_features.shape[-1]))
            train_mask[train_mask_ids] += 1
            train_mask = train_mask.to(self.device)
        
        if train_node_ids is not None:
            node_interact_times = torch.Tensor([min_time]).broadcast_to(train_node_ids.shape).numpy()
            past_node_embeddings = self.aggregated_node_embeddings(node_ids = train_node_ids, 
                                                               node_interact_times = node_interact_times,
                                                               num_neighbors = num_neighbors, 
                                                               time_gap = time_gap,
                                                               full = False)
        else:
            node_ids = torch.arange(self.node_raw_features.shape[0]).to(torch.int64).numpy()
            node_interact_times = torch.Tensor([min_time]).broadcast_to(node_ids.shape).numpy()
            past_node_embeddings = self.aggregated_node_embeddings(node_ids = node_ids, 
                                                                node_interact_times = node_interact_times,
                                                                num_neighbors = num_neighbors, 
                                                                time_gap = time_gap,
                                                                full = True)
        
        spatial_node_embeddings = torch.zeros(self.node_raw_features.shape).to(self.device)
        if train_node_ids is not None:
            spatial_node_embeddings[train_node_ids] += past_node_embeddings
        else:
            spatial_node_embeddings = past_node_embeddings    
        
        # spatial_node_embeddings = torch.clone(self.node_raw_features)
    
        if pe is not None:
            spatial_node_embeddings += pe

        spatial_node_embeddings = self.transformer(spatial_node_embeddings.unsqueeze(0)).squeeze(0)
        
        if train_mask is not None:
            spatial_node_embeddings *= train_mask
            
        return spatial_node_embeddings
    
    # def compute_node_embeddings(self, pe, node_ids, node_interact_times, num_neighbors, time_gap):
    #     node_embeddings = self.aggregated_node_embeddings(node_ids, node_interact_times, num_neighbors, time_gap)
    #     node_embeddings = self.linear_transformer(pe, node_ids, node_embeddings)

    #     return node_embeddings

    
    ###* NEW WAY of computing temporal node embeddings
    ### Combining the node embedding learned by Graph Transformer (spatial_node_embeddings)
    ### and the temporal information of a node u at current time (sample recent links, recent neighbors features)
    ### to obtain final representation for node u
    def compute_node_embeddings(self, spatial_node_embeddings, node_ids, node_interact_times, num_neighbors, time_gap):
        batch_temporal_node_embeddings = self.aggregated_node_embeddings(node_ids, node_interact_times, num_neighbors, time_gap)
        batch_spatial_node_embeddings = spatial_node_embeddings[node_ids]
        
        # node_embeddings = batch_temporal_node_embeddings + batch_spatial_node_embeddings

        node_embeddings = torch.concat([batch_temporal_node_embeddings, batch_spatial_node_embeddings], dim = -1)
        for layer in self.node_embeddings_mlps:
            node_embeddings = layer(node_embeddings)

        return node_embeddings

        # return batch_temporal_node_embeddings + batch_spatial_node_embeddings