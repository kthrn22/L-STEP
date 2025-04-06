import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.sparse as sparse
import torch.fft as fft
import pdb
# from linformer import Linformer
from models.modules import TimeEncoder
from torch_scatter import scatter, scatter_mean

            
def modify_tensor(source_tensor, ids, new_tensor):
    '''
        all params are tensor type
        ids: np array
    '''

    device = source_tensor.device

    new_source_tensor = torch.clone(source_tensor)
    new_source_tensor.scatter_(dim = 0, 
                               index = torch.from_numpy(ids).unsqueeze(-1).broadcast_to(new_tensor.shape).to(device), 
                               src = new_tensor)

    return new_source_tensor

class LSTEP(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler, full_neighbor_sampler, pe_dim: int,
                 num_neighbors: int, time_feat_dim: int, num_fft_batches: int, use_dropout = False, dropout: float = 0.1, weighted_sum = False,
                 concat_pe = True, device: str = 'cuda'):
        
        super(LSTEP, self).__init__()

        edge_feat_dim = edge_raw_features.shape[-1]
        node_feat_dim = node_raw_features.shape[-1]
        self.num_fft_batches = num_fft_batches 
        self.num_nodes = node_raw_features.shape[0]
        self.pe_dim = pe_dim
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.concat_pe = concat_pe
        # self.positional_encoding = 
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.full_neighbor_sampler = full_neighbor_sampler
        self.time_encoder = TimeEncoder(time_feat_dim, parameter_requires_grad = False)
        self.device = device

        # self.fft_filter = nn.Linear(pe_dim, pe_dim, bias = False).to(torch.complex64)
        self.fft_filter = nn.Linear(pe_dim, num_fft_batches, bias = False).to(torch.complex64)
        self.fft_dropout = nn.Dropout(p = dropout)
        self.fft_agg = nn.Linear(num_fft_batches, 1, bias = False)

        self.edge_mlp_1 = nn.Linear(edge_feat_dim + time_feat_dim, edge_feat_dim + time_feat_dim)
        self.edge_agg = nn.Linear(num_neighbors, 1)
        self.edge_mlp_2 = nn.Linear(edge_feat_dim + time_feat_dim, edge_feat_dim + time_feat_dim)

        self.node_mlp = nn.Linear(edge_feat_dim + node_feat_dim + time_feat_dim, node_feat_dim)

        self.self_update_pe = nn.Linear(pe_dim, pe_dim)
        self.pe_mlp_1 = nn.Linear(pe_dim + time_feat_dim, pe_dim)
        self.pe_mlp_2 = nn.Linear(pe_dim, pe_dim)

        self.self_update_neighbor_pe = nn.Linear(pe_dim, pe_dim)
        self.pe_neighbor_mlp_1 = nn.Linear(pe_dim + time_feat_dim, pe_dim)
        self.pe_neighbor_mlp_2 = nn.Linear(pe_dim, pe_dim)

        self.out_node_emb = nn.Linear(pe_dim + node_feat_dim, node_feat_dim)

        self.weighted_sum = weighted_sum

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

    def ablation_fourier_transform_pe(self, node_ids, pe, batch_idx, use_dropout = False, use_mixer = False):
        batch_pe = pe[torch.from_numpy(node_ids)]       # [batch_size x batch_idx x pe_dim]
        mask = None
        
        if batch_pe.shape[1] < self.num_fft_batches:
            padding = torch.zeros((batch_pe.shape[0], self.num_fft_batches - batch_pe.shape[1], batch_pe.shape[-1])).to(self.device)
            batch_pe = torch.cat([batch_pe, padding], dim = 1) # [batch_size x T x pe_dim]
        
            mask = torch.zeros(batch_pe.shape).to(self.device)
            mask[:, : batch_idx, :] += 1                                  
        
        init_pe = torch.clone(batch_pe)
        
        current_pe = self.fft_agg(batch_pe.permute(0, 2, 1)).squeeze()     # [N x num_fft_batches x pe_dim] -> [N x pe_dim x num_fft_batches] -> [N x pe_dim]
       
        return current_pe                        

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
        
        batch_pe = self.fft_filter.weight.unsqueeze(0) * batch_pe
        if mask is not None:
            batch_pe *= mask
        
        batch_pe = fft.ifftn(batch_pe, dim = 1)               # [N x T x pe_dim]
        if mask is not None:
            batch_pe *= mask
        
        batch_pe = batch_pe.to(torch.float32)
        
        if use_dropout:
            batch_pe = self.fft_dropout(batch_pe)
            batch_pe += init_pe

        current_pe = self.fft_agg(batch_pe.permute(0, 2, 1)).squeeze()     # [N x num_fft_batches x pe_dim] -> [N x pe_dim x num_fft_batches] -> [N x pe_dim]
       
        return current_pe                                                       # [N x pe_dim]
    
    def aggregated_node_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
        num_neighbors: int = 20, time_gap: int = 2000, testing = False):
        '''
        obtain node & edge embeddings
        return aggregated node embeddings for each node
        '''
        # get links
        # if full is False:
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
                                                        node_interact_times = node_interact_times,
                                                        num_neighbors = num_neighbors)
    
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        
        ### process edge_feat and time_feat
        # (N, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = torch.cat([nodes_neighbor_time_features, nodes_edge_raw_features], dim=-1)
        init_combined_features = torch.clone(combined_features)
        # (N, num_neighbors, edge_feat_dim + time_feat_dim) -> (N, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = self.edge_mlp_1(combined_features)
        # (N, num_neighbors, edge_feat_dim + time_feat_dim) -> (N, edge_feat_dim + time_feat_dim, num_neighbors)
        # -> (N, edge_feat_dim + time_feat_dim, 1) -> (N, edge_feat_dim + time_feat_dim)
        combined_features = self.edge_agg(combined_features.permute(0, 2, 1)).squeeze()
        # (N, edge_feat_dim + time_feat_dim)
        combined_features = f.relu(combined_features)
        # (N, edge_feat_dim + time_feat_dim)
        # combined_features = f.layer_norm(combined_features, normalized_shape = combined_features.shape[-1])
        # (N, edge_feat_dim + time_feat_dim)
        combined_features = self.edge_mlp_2(combined_features)
        if self.use_dropout:
            combined_features = f.dropout(combined_features, p = self.dropout)
        # (N, edge_feat_dim + time_feat_dim)
        # combined_features += init_combined_features.sum(dim = 1)
        ###

        time_gap_neighbor_node_ids, _, time_gap_neighbor_times = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                        node_interact_times=node_interact_times,
                                                                                        num_neighbors=time_gap)    
        
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(time_gap_neighbor_node_ids)]
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)
        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        # nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)
        
        if self.weighted_sum:
            unique_times, inv_ids = torch.unique(torch.from_numpy(time_gap_neighbor_times), return_inverse = True)
            offset_inv_ids = inv_ids + (torch.arange(inv_ids.shape[0]) * unique_times.shape[0]).unsqueeze(-1)
            batch_unique_times = torch.zeros(inv_ids.shape[0], unique_times.shape[0]).flatten()
            batch_unique_times = scatter_mean(src = torch.from_numpy(time_gap_neighbor_times).flatten(), 
                                            index = offset_inv_ids.flatten(),
                                            out = batch_unique_times,
                                            dim = -1)
            
            batch_unique_times = batch_unique_times.view(inv_ids.shape[0], unique_times.shape[0])
            weights = torch.exp(-(torch.from_numpy(node_interact_times).unsqueeze(-1) - batch_unique_times)) * (batch_unique_times != 0.0)
            sum_weights = torch.sum(weights, dim = -1)
            sum_weights = sum_weights + (sum_weights == 0)
            weights = weights / (sum_weights.unsqueeze(-1))
            weights = weights.flatten()[offset_inv_ids.flatten()].view(time_gap_neighbor_times.shape).clamp(0, 1).to(torch.float32)

            nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1) * weights.to(self.device).unsqueeze(-1), dim=1)
        else:
            nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)
        
        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]
        # Tensor, shape (batch_size, node_feat_dim)
        # node_embeddings = self.node_final_mlp(torch.cat([combined_features, output_node_features], dim=1))

        # (N, node_feat_dim + edge_feat_dim + time_feat_dim)
        if testing:
            pdb.set_trace()

        node_embeddings = self.node_mlp(torch.cat([output_node_features, combined_features], dim = -1))
        return node_embeddings
    
    def compute_neighborhood_pe(self, pe, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 30):
        neighbor_node_ids, _, neighbor_times = self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids, 
                                                                                              node_interact_times = node_interact_times, 
                                                                                              num_neighbors = num_neighbors)

        # [B, num_neighbors] (B: batch_size)
        time_diff = torch.from_numpy(node_interact_times).unsqueeze(-1) - torch.from_numpy(neighbor_times)
        # [B, num_neighbors, time_feat_dim]
        time_features = self.time_encoder(time_diff.float().to(self.device))
        time_features[torch.from_numpy(neighbor_node_ids) == 0] = 0.0
        # [B, num_neighbors, pe_dim]
        neighbor_pe = pe[torch.from_numpy(neighbor_node_ids)]
        # [B, pe_dim]
        node_pe = pe[torch.from_numpy(node_ids)]

        # [B, num_neighbors, pe_dim + time_feat_dim] -> [B, pe_dim + time_feat_dim]
        neighbor_aggregated_pe = torch.cat([neighbor_pe, time_features], dim = -1).sum(dim = 1)
        # [B, pe_dim + time_feat_dim] -> [B, pe_dim]
        neighbor_aggregated_pe = self.pe_neighbor_mlp_1(neighbor_aggregated_pe)
        neighbor_aggregated_pe = f.relu(neighbor_aggregated_pe)
        # [B, pe_dim] -> [B, pe_dim]
        neighbor_aggregated_pe = self.pe_neighbor_mlp_2(neighbor_aggregated_pe)
        neighbor_aggregated_pe = self.self_update_neighbor_pe(node_pe) + neighbor_aggregated_pe
        neighbor_aggregated_pe = f.tanh(neighbor_aggregated_pe)

        neighbor_aggregated_pe = node_pe + neighbor_aggregated_pe

        return neighbor_aggregated_pe

    def combining_pe_raw_feat(self, pe, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 30, 
                              time_gap: int = 2000, testing = False):
        aggregated_node_embedding = self.aggregated_node_embeddings(node_ids = node_ids,
                                                                    node_interact_times = node_interact_times,
                                                                    num_neighbors = num_neighbors,
                                                                    time_gap = time_gap,
                                                                    testing = testing)
        
        # return aggregated_node_embedding
        
        neighbor_aggregated_pe = self.compute_neighborhood_pe(pe, node_ids = node_ids,
                                                            node_interact_times = node_interact_times, num_neighbors = num_neighbors)
        
        out_node_emb = self.out_node_emb(torch.cat([aggregated_node_embedding, neighbor_aggregated_pe], dim = -1))

        return out_node_emb

    def update_pe(self, pe, node_ids: np.ndarray, edge_ids: np.ndarray, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray, node_interact_times: np.ndarray, current_time, 
                                         num_neighbors: int = 30, time_gap: int = 2000):
        # (N, node_feat_dim)
        # node_feat = self.node_raw_features[torch.from_numpy(node_ids)]
        # (N, pe_dim)
        node_pe = pe[torch.from_numpy(node_ids)]
        # (E, edge_feat_dim)
        # edge_feat = self.edge_raw_features[torch.from_numpy(edge_ids)]
        # (E, time_feat_dim)
        time_diff = (torch.Tensor([current_time]).broadcast_to(node_interact_times.shape) - torch.from_numpy(node_interact_times)).float().to(self.device)
        # (E, time_feat_dim)
        time_features = self.time_encoder(time_diff.unsqueeze(-1)).squeeze(1)
        
        #### update pe
        aggregated_pe = torch.zeros((pe.shape[0], pe.shape[1] + time_features.shape[1])).to(self.device)
        scatter(src = torch.cat([pe[batch_dst_node_ids], time_features], dim = -1), 
                index = torch.from_numpy(batch_src_node_ids).to(self.device), 
                dim = 0, out = aggregated_pe, 
                reduce = "sum")
        scatter(src = torch.cat([pe[batch_src_node_ids], time_features], dim = -1), 
                index = torch.from_numpy(batch_dst_node_ids).to(self.device), 
                dim = 0, out = aggregated_pe, 
                reduce = "sum")
        
        aggregated_pe = aggregated_pe[node_ids]
        # [N, pe_dim + time_feat_dim] -> [N, pe_dim]
        aggregated_pe = self.pe_mlp_1(aggregated_pe)
        aggregated_pe = f.relu(aggregated_pe)
        # [N, pe_dim] -> [N, pe_dim]
        aggregated_pe = self.pe_mlp_2(aggregated_pe)
        # [N, pe_dim]
        updated_pe =  self.self_update_pe(node_pe) + aggregated_pe
        updated_pe = f.tanh(updated_pe)
        updated_pe = node_pe + updated_pe

        pe[node_ids] = updated_pe
        
        ### updated not involved pe
        neighbor_node_ids, _, neighbor_times = self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
                                                    node_interact_times = node_interact_times,
                                                    num_neighbors = num_neighbors)
        
        src_node_ids = torch.from_numpy(node_ids).unsqueeze(-1).broadcast_to(neighbor_node_ids.shape).flatten()
        neighbor_node_ids = neighbor_node_ids.flatten()
        neighbor_times = neighbor_times.flatten()

        time_diff = (torch.Tensor([current_time]).broadcast_to(neighbor_times.shape) - torch.from_numpy(neighbor_times)).float().to(self.device)
        time_features = self.time_encoder(time_diff.unsqueeze(-1)).squeeze(1)
        time_features[torch.from_numpy(neighbor_node_ids) == 0] = 0.0
        pe[0] = 0.0

        neighbor_aggregated_pe = torch.zeros((pe.shape[0], pe.shape[1] + time_features.shape[1])).to(self.device)
        scatter(src = torch.cat([pe[src_node_ids], time_features], dim = -1), 
                index = torch.from_numpy(neighbor_node_ids).to(self.device),
                dim = 0, out = neighbor_aggregated_pe)
        
        neighbor_aggregated_pe = neighbor_aggregated_pe[torch.from_numpy(neighbor_node_ids).unique()]
        node_pe = pe[torch.from_numpy(neighbor_node_ids).unique()]
        
        ## self update with MLP
        # [N, pe_dim + time_feat_dim] -> [N, pe_dim, pe_dim]
        neighbor_aggregated_pe = self.pe_mlp_1(neighbor_aggregated_pe)
        neighbor_aggregated_pe = f.relu(neighbor_aggregated_pe)
        # [N, pe_dim] -> [N, pe_dim]
        neighbor_aggregated_pe = self.pe_mlp_2(neighbor_aggregated_pe)
        # [N, pe_dim]
        updated_pe = self.self_update_pe(node_pe) + neighbor_aggregated_pe
        updated_pe = f.tanh(neighbor_aggregated_pe)
        updated_pe = node_pe + updated_pe
        ##
        
        pe[torch.from_numpy(neighbor_node_ids).unique()] = updated_pe

        return pe