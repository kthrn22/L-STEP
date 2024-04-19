import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import math
import scipy
from models.modules import TimeEncoder
from models.TempGT import TempGT
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
# from evaluate_models_utils import evaluate_model_link_prediction
from evaluate_TempGT_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.PositionalEncoding import LaplacianPE 

class TempGNTK(object):
    def __init__(self, num_mlp_layers = 2):
        self.num_mlp_layers = num_mlp_layers

    def __next_diag(self, S):
        diag = torch.sqrt(S.diag())
        S /= (diag[:, None] * diag[None, :])
        S = torch.clamp(S, -1, 1)

        dS = (math.pi - torch.arccos(S)) / (2 * math.pi)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / (2 * math.pi)
        S *= (diag[:, None] * diag[None, :])

        return S, dS, diag

    def normalize_length(self, X):
        r"""
            Args:
                X (torch.tensor): shape [N, K, d]

            Returns:
                X (torch.tensor): vector length is normalized to 1 in the last dimension
        """
        # X: [N, K, d] -> length [N, K, 1]
        length = torch.sqrt(torch.sum(X ** 2, dim = -1)).unsqueeze(-1)
        length += (length == 0)
        # scale vector to length of 1 -> [N, K, d]
        return X / length

    def get_diag_list(self, node_emb, A, return_ntk = False, normalize_length_emb = False):
        n = node_emb.shape[0]
        if normalize_length_emb:
            node_emb = self.normalize_length(node_emb)
        
        sigma = torch.mm(node_emb, node_emb.T).nan_to_num()

        if A is not None:
            if n > 1000:
                sparse_A = A.to_sparse_coo()
                row, col = sparse_A.indices()
                vals = sparse_A.values()
                sparse_A = scipy.sparse.coo_array((vals, (row, col)), shape = (n, n))

                adj_block = np.nan_to_num(scipy.sparse.kron(sparse_A, sparse_A)).astype(np.float64)
                sigma = np.nan_to_num(adj_block.dot(sigma.view(-1, 1).numpy()))
                sigma = torch.from_numpy(sigma).view(n, n)

            else:
                adj_block = torch.kron(A, A).nan_to_num()
                sigma = torch.mm(adj_block.to(torch.float), sigma.view(-1, 1)).view(n, n)
                sigma = sigma.nan_to_num()
        
        ntk = torch.clone(sigma)
        sigma = sigma
        
        diag_list = []
        for _ in range(self.num_mlp_layers):
            sigma, dot_sigma, diag = self.__next_diag(sigma)
            sigma = sigma.nan_to_num()
            dot_sigma = dot_sigma.nan_to_num()
            diag = diag.nan_to_num()
            ntk = ntk * dot_sigma + sigma
            ntk = ntk.nan_to_num()
            diag_list.append(diag)

        if return_ntk:
            return ntk, diag_list

        return diag_list

class LinkPredictor(object):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 num_neighbors: int, time_feat_dim: int, num_edge_layers: int = 2, device: str = 'cuda'):
        edge_feat_dim = edge_raw_features.shape[-1]
        node_feat_dim = node_raw_features.shape[-1]
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = TimeEncoder(time_feat_dim, parameter_requires_grad = False).to(device)
        self.device = device

        self.temp_gntk = TempGNTK()

        self.edge_mlp_layers = nn.ModuleList(
            nn.Linear(time_feat_dim + edge_feat_dim, time_feat_dim + edge_feat_dim) for _ in range(num_edge_layers)
        )
        self.edge_mlp = nn.Linear(edge_feat_dim + time_feat_dim, edge_feat_dim)
        self.edge_final_mlp = nn.Linear(num_neighbors, 1)

        self.node_final_mlp = nn.Linear(edge_feat_dim + node_feat_dim, node_feat_dim)

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

    def aggregated_node_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
        num_neighbors: int = 20, time_gap: int = 2000):
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids = node_ids,
                                                           node_interact_times = node_interact_times,
                                                           num_neighbors = num_neighbors)

        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        
        # (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        edge_embeddings = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        edge_embeddings = torch.mean(edge_embeddings, dim = 1)
        ####
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
        node_embeddings = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]

        return node_embeddings, edge_embeddings
    
    def compute_self_gntk(self, node_embeddings, edge_embeddings, return_normalize = True):
        node_ntk, node_emb_diag_list = self.temp_gntk.get_diag_list(node_embeddings, A = None, return_ntk = True)
        edge_ntk, edge_emb_diag_list = self.temp_gntk.get_diag_list(edge_embeddings, A = None, return_ntk = True)     

        return node_ntk, node_emb_diag_list, edge_ntk, edge_emb_diag_list   
    
    def compute_gntk(self, node_embeddings_1, node_embeddings_2, agg_edge_embeddings_1, agg_edge_embeddings_2):
        pass    

def evaluate_ntk(model, gram_matrix, neighbor_sampler, evaluate_idx_data_loader,
                evaluate_neg_edge_sampler, evaluate_data, loss_func: nn.Module,
                num_neighbors: int = 20, time_gap: int = 2000):
    
    model.set_neighbor_sampler(neighbor_sampler)

    evaluate_losses, evaluate_metrics = [], []
    evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
    for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
        evaluate_data_indices = evaluate_data_indices.numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
            evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
            evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

        if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
            batch_neg_src_node_ids, batch_neg_dst_node_ids = val_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                batch_src_node_ids=batch_src_node_ids,
                                                                                                batch_dst_node_ids=batch_dst_node_ids,
                                                                                                current_batch_start_time=batch_node_interact_times[0],
                                                                                                current_batch_end_time=batch_node_interact_times[-1])
        else:
            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

        # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
        # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            
        
        # assert positional_encoding is not None                

        # batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
        # batch_node_ids = np.array(list(set(batch_node_ids)))
        # # (batch_size, pe_dim)
        # fft_current_positional_encoding = model[0].fourier_transform_pe(batch_node_ids, positional_encoding, batch_idx)
        # current_positional_encoding = torch.clone(positional_encoding[:, batch_idx - 1, :])
        # current_positional_encoding[torch.from_numpy(batch_node_ids)] = fft_current_positional_encoding

        batch_src_node_embeddings, batch_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_src_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)

        batch_dst_node_embeddings, batch_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_dst_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)
        
        batch_neg_src_node_embeddings, batch_neg_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_src_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)
        
        batch_neg_dst_node_embeddings, batch_neg_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_dst_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)    

        # get positive and negative probabilities, shape (batch_size, )
        positive_probabilities = (gram_matrix[batch_src_node_ids, batch_dst_node_ids]).clamp(0, 1)
        negative_probabilities = (gram_matrix[batch_neg_src_node_ids, batch_neg_dst_node_ids]).clamp(0, 1)
       
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

        loss = loss_func(input=predicts, target=labels)

        evaluate_losses.append(loss.item())

        evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

        evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        return evaluate_losses, evaluate_metrics

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    model = LinkPredictor(node_raw_features, edge_raw_features, train_neighbor_sampler, args.num_neighbors, args.time_feat_dim)
    n = node_raw_features.shape[0]
    gram_matrix = torch.zeros((n, n)).to(args.device)

    node_embeddings = torch.zeros((node_raw_features.shape)).to(args.device)
    agg_edge_embeddings = torch.zeros((edge_raw_features.shape)).to(args.device)
    
    loss_func = nn.BCELoss()
    # model = convert_to_gpu(model, device = args.device)
    train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
    for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
        train_data_indices = train_data_indices.numpy()
        
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
            train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
            train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

        _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
        batch_neg_src_node_ids = batch_src_node_ids

        # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
        # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
        
        batch_src_node_embeddings, batch_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_src_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)

        batch_dst_node_embeddings, batch_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_dst_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)
        
        node_embeddings[torch.from_numpy(batch_src_node_ids)] = batch_src_node_embeddings
        node_embeddings[torch.from_numpy(batch_dst_node_ids)] = batch_dst_node_embeddings
        # agg_edge_embeddings[torch.from_numpy(batch_src_node_ids)] = batch_src_edge_embeddings
        # agg_edge_embeddings[torch.from_numpy(batch_dst_node_ids)] = batch_dst_edge_embeddings
        
        batch_neg_src_node_embeddings, batch_neg_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_src_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)
        
        batch_neg_dst_node_embeddings, batch_neg_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_dst_node_ids,
                                                                                                node_interact_times = batch_node_interact_times,
                                                                                                num_neighbors = args.num_neighbors,
                                                                                                time_gap = args.time_gap)    

        train_idx_data_loader_tqdm.set_description(f'computing NTK...')
        node_ntk, node_emb_diag_list, edge_ntk, edge_emb_diag_list = model.compute_self_gntk(batch_src_node_embeddings, batch_src_edge_embeddings)
        batch_ntk = node_ntk + edge_ntk

        gram_matrix[torch.from_numpy(batch_src_node_ids).unsqueeze(-1), batch_dst_node_ids] = batch_ntk
        gram_matrix[torch.from_numpy(batch_dst_node_ids).unsqueeze(-1), batch_src_node_ids] = batch_ntk   

    #### evaluate set ####
    ### normalize gram matrix here
    # gram_matrix /= gram_matrix.min()
    # gram_matrix = gram_matrix.nan_to_num()
    val_losses, val_metrics = evaluate_ntk(model, gram_matrix, neighbor_sampler = full_neighbor_sampler, evaluate_idx_data_loader = val_idx_data_loader,
                                        evaluate_neg_edge_sampler = val_neg_edge_sampler, evaluate_data = val_data, 
                                        loss_func = nn.BCELoss(), num_neighbors = args.num_neighbors, time_gap = args.time_gap)

    new_node_val_losses, new_node_val_metrics = evaluate_ntk(model, gram_matrix, neighbor_sampler = full_neighbor_sampler, evaluate_idx_data_loader = new_node_val_idx_data_loader,
                                        evaluate_neg_edge_sampler = new_node_val_neg_edge_sampler, evaluate_data = new_node_val_data, 
                                        loss_func = nn.BCELoss(), num_neighbors = args.num_neighbors, time_gap = args.time_gap)
    
    test_losses, test_metrics = evaluate_ntk(model, gram_matrix, neighbor_sampler = full_neighbor_sampler, evaluate_idx_data_loader = test_idx_data_loader,
                                        evaluate_neg_edge_sampler = test_neg_edge_sampler, evaluate_data = test_data, 
                                        loss_func = nn.BCELoss(), num_neighbors = args.num_neighbors, time_gap = args.time_gap)

    new_node_test_losses, new_node_test_metrics = evaluate_ntk(model, gram_matrix, neighbor_sampler = full_neighbor_sampler, evaluate_idx_data_loader = new_node_test_idx_data_loader,
                                        evaluate_neg_edge_sampler = new_node_test_neg_edge_sampler, evaluate_data = new_node_test_data, 
                                        loss_func = nn.BCELoss(), num_neighbors = args.num_neighbors, time_gap = args.time_gap)
        
    val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

    print(f'validate loss: {np.mean(val_losses):.4f}')
    for metric_name in val_metrics[0].keys():
        average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
        
        print(f'validate {metric_name}, {average_val_metric:.4f}')
        val_metric_dict[metric_name] = average_val_metric

    # logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
    print(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
    for metric_name in new_node_val_metrics[0].keys():
        average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
        # logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
        print(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
        new_node_val_metric_dict[metric_name] = average_new_node_val_metric

    # logger.info(f'test loss: {np.mean(test_losses):.4f}')
    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
        # logger.info(f'test {metric_name}, {average_test_metric:.4f}')
        print(f'test {metric_name}, {average_test_metric:.4f}')
        test_metric_dict[metric_name] = average_test_metric

    # logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    print(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    for metric_name in new_node_test_metrics[0].keys():
        average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        # logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        print(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        new_node_test_metric_dict[metric_name] = average_new_node_test_metric
    #### test set ####
    



    # for run in range(args.num_runs):
    #     for epoch in range(args.num_epochs):
    #         # store train losses and metrics
    #         train_losses, train_metrics = [], []
    #         train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
    #         last_batch_idx = len(train_idx_data_loader_tqdm) - 1
    #         for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm)
    #             train_data_indices = train_data_indices.numpy()
                
    #             batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
    #                 train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
    #                 train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

    #             _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
    #             batch_neg_src_node_ids = batch_src_node_ids

    #             # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
    #             # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                
    #             batch_src_node_embeddings, batch_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_src_node_ids,
    #                                                                                                     node_interact_times = batch_node_interact_times,
    #                                                                                                     num_neighbors = args.num_neighbors,
    #                                                                                                     time_gap = args.time_gap)

    #             batch_dst_node_embeddings, batch_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_dst_node_ids,
    #                                                                                                     node_interact_times = batch_node_interact_times,
    #                                                                                                     num_neighbors = args.num_neighbors,
    #                                                                                                     time_gap = args.time_gap)
                
    #             batch_neg_src_node_embeddings, batch_neg_src_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_src_node_ids,
    #                                                                                                     node_interact_times = batch_node_interact_times,
    #                                                                                                     num_neighbors = args.num_neighbors,
    #                                                                                                     time_gap = args.time_gap)
                
    #             batch_neg_dst_node_embeddings, batch_neg_dst_edge_embeddings = model.aggregated_node_embeddings(node_ids = batch_neg_dst_node_ids,
    #                                                                                                     node_interact_times = batch_node_interact_times,
    #                                                                                                     num_neighbors = args.num_neighbors,
    #                                                                                                     time_gap = args.time_gap)
                
                # get positive and negative probabilities, shape (batch_size, )
                # positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                # predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                # labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)


                # train_losses.append(lp_loss.item())

                # train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                # train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, lp loss: {lp_loss.item()}, pe loss: {pe_loss.item()}')
    
    
    # for run in range(args.num_runs):

    #     set_random_seed(seed=run)

    #     args.seed = run
    #     args.save_model_name = f'{args.model_name}_seed{args.seed}'
    #     args.save_trained_pe = f'{args.model_name}_pe_seed{args.seed}'
        
    #     # set up logger
    #     logging.basicConfig(level=logging.INFO)
    #     logger = logging.getLogger()
    #     logger.setLevel(logging.DEBUG)
    #     os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
    #     # create file handler that logs debug and higher level messages
    #     fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
    #     fh.setLevel(logging.DEBUG)
    #     # create console handler with a higher log level
    #     ch = logging.StreamHandler()
    #     ch.setLevel(logging.WARNING)
    #     # create formatter and add it to the handlers
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     fh.setFormatter(formatter)
    #     ch.setFormatter(formatter)
    #     # add the handlers to logger
    #     logger.addHandler(fh)
    #     logger.addHandler(ch)

    #     run_start_time = time.time()
    #     logger.info(f"********** Run {run + 1} starts. **********")

    #     logger.info(f'configuration is {args}')

    #     #### define dynamic backbone
    #     dynamic_backbone = TempGT(node_raw_features = node_raw_features,
    #                               edge_raw_features = edge_raw_features,
    #                               neighbor_sampler = train_neighbor_sampler,
    #                               pe_dim = args.position_feat_dim,
    #                               num_neighbors = args.num_neighbors,
    #                               time_feat_dim = args.time_feat_dim,
    #                               num_batches = len(train_idx_data_loader),
    #                               seq_len = args.batch_size,
    #                               num_heads = args.num_heads,
    #                               transformer_depth = args.transformer_depth,
    #                               num_edge_layers = 2,
    #                               dropout = args.dropout,
    #                               transformer = "linformer", 
    #                               device = args.device)
        
    #     # define positional encoding and transformer model

    #     link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
    #                                 hidden_dim=node_raw_features.shape[1], output_dim=1)
    #     model = nn.Sequential(dynamic_backbone, link_predictor)

    #     if args.pre_trained == 'True':
    #         logger.info(f'Loading previous best model...')
    #         model.load_state_dict(torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_seed{}.pkl".format(args.dataset_name, run, run), map_location = None))
    #         # final_trained_positional_encoding = torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_pe_seed{}.pkl".format(args.dataset_name, run, run)).to(args.device)

    #     logger.info(f'model -> {model}')
    #     logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
    #                 f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

    #     optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    #     model = convert_to_gpu(model, device=args.device)

    #     save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    #     shutil.rmtree(save_model_folder, ignore_errors=True)
    #     os.makedirs(save_model_folder, exist_ok=True)

    #     early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
    #                                    save_model_name=args.save_model_name, save_trained_pe = args.save_trained_pe, 
    #                                    logger=logger, model_name=args.model_name)

    #     pe_loss_func = nn.MSELoss()
    #     loss_func = nn.BCELoss()

    #     # compute initial_positional_encoding 
    #     num_nodes = node_raw_features.shape[0]
    #     train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
    #     for _, train_data_indices in enumerate(train_idx_data_loader_tqdm):
    #         train_data_indices = train_data_indices.numpy()

    #         batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
    #             train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
    #             train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]            

    #         edge_index = torch.from_numpy(np.array([batch_src_node_ids.tolist() + batch_dst_node_ids.tolist(),
    #                                                 batch_dst_node_ids.tolist() + batch_src_node_ids.tolist()]))
    #         break 

    #     # [N, pe_dim]
    #     k = min(num_nodes, args.position_feat_dim)
    #     initial_positional_encoding = LaplacianPE(edge_index, num_nodes, k)
    #     initial_positional_encoding = initial_positional_encoding.to(args.device)

    #     for epoch in range(args.num_epochs):
    #         # initalize postional encoding
    #         # positional_encoding = torch.zeros((num_nodes, len(train_idx_data_loader), args.position_feat_dim)).to(args.device)

    #         positional_encoding = torch.tensor([])
            
    #         model.train()
    #         torch.autograd.set_detect_anomaly(True)
    #         # store train losses and metrics
    #         train_losses, train_metrics = [], []
    #         train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
    #         last_batch_idx = len(train_idx_data_loader_tqdm) - 1
    #         for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
    #             positional_encoding = positional_encoding.to(args.device)
    #             train_data_indices = train_data_indices.numpy()
                
    #             batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
    #                 train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
    #                 train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

    #             _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
    #             batch_neg_src_node_ids = batch_src_node_ids

    #             # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
    #             # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                
    #             if batch_idx == 0:
    #                 current_positional_encoding = None
    #             else:
    #                 batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
    #                 batch_node_ids = np.array(list(set(batch_node_ids)))
    #                 # (batch_size, pe_dim)
    #                 fft_current_positional_encoding = model[0].fourier_transform_pe(batch_node_ids, positional_encoding, batch_idx)
    #                 current_positional_encoding = torch.clone(positional_encoding[:, batch_idx - 1, :])
    #                 current_positional_encoding[torch.from_numpy(batch_node_ids)] = fft_current_positional_encoding

            
    #             batch_src_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
    #                                                                          node_ids = batch_src_node_ids, 
    #                                                                          node_interact_times = batch_node_interact_times, 
    #                                                                          num_neighbors = args.num_neighbors, 
    #                                                                          time_gap = args.time_gap)
                

    #             batch_dst_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
    #                                                                          node_ids = batch_dst_node_ids, 
    #                                                                          node_interact_times = batch_node_interact_times, 
    #                                                                          num_neighbors = args.num_neighbors, 
    #                                                                          time_gap = args.time_gap)
                
    #             batch_neg_src_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
    #                                                                          node_ids = batch_neg_src_node_ids, 
    #                                                                          node_interact_times = batch_node_interact_times, 
    #                                                                          num_neighbors = args.num_neighbors, 
    #                                                                          time_gap = args.time_gap)
                
    #             batch_neg_dst_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
    #                                                                          node_ids = batch_neg_dst_node_ids, 
    #                                                                          node_interact_times = batch_node_interact_times, 
    #                                                                          num_neighbors = args.num_neighbors, 
    #                                                                          time_gap = args.time_gap)
                
    #             # get positive and negative probabilities, shape (batch_size, )
    #             positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
    #             negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

    #             predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
    #             labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

    #             lp_loss = loss_func(input=predicts, target=labels)

    #             # get similarity of positional encodings of positive and negative links and compute PE loss
    #             # (batch_size, pe_dim)
    #             if current_positional_encoding is not None:
    #                 pos_src_pe, pos_dst_pe = current_positional_encoding[torch.from_numpy(batch_src_node_ids)], \
    #                                         current_positional_encoding[torch.from_numpy(batch_dst_node_ids)]
    #                 neg_src_pe, neg_dst_pe = current_positional_encoding[torch.from_numpy(batch_neg_src_node_ids)], \
    #                                         current_positional_encoding[torch.from_numpy(batch_neg_dst_node_ids)]
    #                 positive_pe_loss = pe_loss_func(pos_src_pe, pos_dst_pe)
    #                 negative_pe_loss = pe_loss_func(neg_src_pe, neg_dst_pe)
    #                 pe_loss = positive_pe_loss - negative_pe_loss
    #             else:
    #                 pe_loss = torch.tensor(0).to(torch.float32).to(args.device)

    #             train_losses.append(lp_loss.item())

    #             train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    #             loss = (1 - args.pe_weight) * lp_loss + args.pe_weight * pe_loss
    
    #             # print("Updating Positional Encoding ...")
    #             if batch_idx == 0:
    #                 positional_encoding = torch.cat([positional_encoding, initial_positional_encoding.unsqueeze(1)], dim = 1)
    #             else:
    #                 positional_encoding = torch.cat([positional_encoding, current_positional_encoding.unsqueeze(1)], dim = 1)
                
    #             if current_positional_encoding is not None and batch_idx == last_batch_idx:
    #                 current_positional_encoding = current_positional_encoding.detach().cpu()
    #             positional_encoding = positional_encoding.detach().cpu()

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, lp loss: {lp_loss.item()}, pe loss: {pe_loss.item()}')

    #         # ####### evaluate
    #         final_trained_positional_encoding = current_positional_encoding.to(args.device)

    #         val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                  model=model,
    #                                                                  final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                  neighbor_sampler=full_neighbor_sampler,
    #                                                                  evaluate_idx_data_loader=val_idx_data_loader,
    #                                                                  evaluate_neg_edge_sampler=val_neg_edge_sampler,
    #                                                                  evaluate_data=val_data,
    #                                                                  loss_func=loss_func,
    #                                                                  num_neighbors=args.num_neighbors,
    #                                                                  time_gap=args.time_gap)

          
    #         new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                                    model=model,
    #                                                                                    final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                                    neighbor_sampler=full_neighbor_sampler,
    #                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
    #                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
    #                                                                                    evaluate_data=new_node_val_data,
    #                                                                                    loss_func=loss_func,
    #                                                                                    num_neighbors=args.num_neighbors,
    #                                                                                    time_gap=args.time_gap)

           
    #         logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
    #         for metric_name in train_metrics[0].keys():
    #             logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
    #         logger.info(f'validate loss: {np.mean(val_losses):.4f}')
    #         for metric_name in val_metrics[0].keys():
    #             logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
    #         logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
    #         for metric_name in new_node_val_metrics[0].keys():
    #             logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

    #         # perform testing once after test_interval_epochs
    #         if (epoch + 1) % args.test_interval_epochs == 0:
    #             test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                        model=model,
    #                                                                        final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                        neighbor_sampler=full_neighbor_sampler,
    #                                                                        evaluate_idx_data_loader=test_idx_data_loader,
    #                                                                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
    #                                                                        evaluate_data=test_data,
    #                                                                        loss_func=loss_func,
    #                                                                        num_neighbors=args.num_neighbors,
    #                                                                        time_gap=args.time_gap)

    #             new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                                          model=model,
    #                                                                                          final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                                          neighbor_sampler=full_neighbor_sampler,
    #                                                                                          evaluate_idx_data_loader=new_node_test_idx_data_loader,
    #                                                                                          evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
    #                                                                                          evaluate_data=new_node_test_data,
    #                                                                                          loss_func=loss_func,
    #                                                                                          num_neighbors=args.num_neighbors,
    #                                                                                          time_gap=args.time_gap)

                
    #             logger.info(f'test loss: {np.mean(test_losses):.4f}')
    #             for metric_name in test_metrics[0].keys():
    #                 logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
    #             logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    #             for metric_name in new_node_test_metrics[0].keys():
    #                 logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

    #         # select the best model based on all the validate metrics
    #         val_metric_indicator = []
    #         for metric_name in val_metrics[0].keys():
    #             val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
    #         early_stop = early_stopping.step(val_metric_indicator, model, final_trained_positional_encoding = final_trained_positional_encoding)

    #         if early_stop:
    #             break

    #     ######################### BEST MODEL
    #     # load the best model
    #     early_stopping.load_checkpoint(model)
    #     final_trained_positional_encoding = early_stopping.load_pe()

    #     # evaluate the best model
    #     logger.info(f'get final performance on dataset {args.dataset_name}...')

    #     # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                  model=model,
    #                                                                  final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                  neighbor_sampler=full_neighbor_sampler,
    #                                                                  evaluate_idx_data_loader=val_idx_data_loader,
    #                                                                  evaluate_neg_edge_sampler=val_neg_edge_sampler,
    #                                                                  evaluate_data=val_data,
    #                                                                  loss_func=loss_func,
    #                                                                  num_neighbors=args.num_neighbors,
    #                                                                  time_gap=args.time_gap)

    #         new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                                    model=model,
    #                                                                                    final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                                    neighbor_sampler=full_neighbor_sampler,
    #                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
    #                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
    #                                                                                    evaluate_data=new_node_val_data,
    #                                                                                    loss_func=loss_func,
    #                                                                                    num_neighbors=args.num_neighbors,
    #                                                                                    time_gap=args.time_gap)

    #     test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                model=model,
    #                                                                final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                neighbor_sampler=full_neighbor_sampler,
    #                                                                evaluate_idx_data_loader=test_idx_data_loader,
    #                                                                evaluate_neg_edge_sampler=test_neg_edge_sampler,
    #                                                                evaluate_data=test_data,
    #                                                                loss_func=loss_func,
    #                                                                num_neighbors=args.num_neighbors,
    #                                                                time_gap=args.time_gap)

    #     new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
    #                                                                                  model=model,
    #                                                                                  final_trained_positional_encoding = final_trained_positional_encoding,
    #                                                                                  neighbor_sampler=full_neighbor_sampler,
    #                                                                                  evaluate_idx_data_loader=new_node_test_idx_data_loader,
    #                                                                                  evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
    #                                                                                  evaluate_data=new_node_test_data,
    #                                                                                  loss_func=loss_func,
    #                                                                                  num_neighbors=args.num_neighbors,
    #                                                                                  time_gap=args.time_gap)
    #     # store the evaluation metrics at the current run
    #     val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         logger.info(f'validate loss: {np.mean(val_losses):.4f}')
    #         for metric_name in val_metrics[0].keys():
    #             average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
    #             logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
    #             val_metric_dict[metric_name] = average_val_metric

    #         logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
    #         for metric_name in new_node_val_metrics[0].keys():
    #             average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
    #             logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
    #             new_node_val_metric_dict[metric_name] = average_new_node_val_metric

    #     logger.info(f'test loss: {np.mean(test_losses):.4f}')
    #     for metric_name in test_metrics[0].keys():
    #         average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
    #         logger.info(f'test {metric_name}, {average_test_metric:.4f}')
    #         test_metric_dict[metric_name] = average_test_metric

    #     logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    #     for metric_name in new_node_test_metrics[0].keys():
    #         average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
    #         logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
    #         new_node_test_metric_dict[metric_name] = average_new_node_test_metric

    #     single_run_time = time.time() - run_start_time
    #     logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         val_metric_all_runs.append(val_metric_dict)
    #         new_node_val_metric_all_runs.append(new_node_val_metric_dict)
    #     test_metric_all_runs.append(test_metric_dict)
    #     new_node_test_metric_all_runs.append(new_node_test_metric_dict)

    #     # avoid the overlap of logs
    #     if run < args.num_runs - 1:
    #         logger.removeHandler(fh)
    #         logger.removeHandler(ch)

    #     # save model result
    #     if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #         result_json = {
    #             "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
    #             "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
    #             "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
    #             "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
    #         }
    #     else:
    #         result_json = {
    #             "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
    #             "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
    #         }
    #     result_json = json.dumps(result_json, indent=4)

    #     save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    #     os.makedirs(save_result_folder, exist_ok=True)
    #     save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

    #     with open(save_result_path, 'w') as file:
    #         file.write(result_json)

    # # store the average metrics at the log of the last run
    # logger.info(f'metrics over {args.num_runs} runs:')

    # for metric_name in test_metric_all_runs[0].keys():
    #     logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    #     logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
    #                 f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    # for metric_name in new_node_test_metric_all_runs[0].keys():
    #     logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
    #     logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
    #                 f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    # sys.exit()

