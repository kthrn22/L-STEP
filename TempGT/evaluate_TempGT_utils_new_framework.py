import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_prediction(model_name: str, model: nn.Module, final_trained_positional_encoding: torch.Tensor,
                spatial_node_embeddings: torch.Tensor, 
                neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, num_fft_batches: int = 100,
                num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'TempGT']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
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

            # batch_src_node_embeddings = model[0].compute_node_embeddings(pe = final_trained_positional_encoding, 
            #                                                                 node_ids = batch_src_node_ids, 
            #                                                                 node_interact_times = batch_node_interact_times, 
            #                                                                 num_neighbors = num_neighbors, 
            #                                                                 time_gap = time_gap)
            
            # batch_dst_node_embeddings = model[0].compute_node_embeddings(pe = final_trained_positional_encoding, 
            #                                                                 node_ids = batch_dst_node_ids, 
            #                                                                 node_interact_times = batch_node_interact_times, 
            #                                                                 num_neighbors = num_neighbors, 
            #                                                                 time_gap = time_gap)
            
            # batch_neg_src_node_embeddings = model[0].compute_node_embeddings(pe = final_trained_positional_encoding, 
            #                                                                 node_ids = batch_neg_src_node_ids, 
            #                                                                 node_interact_times = batch_node_interact_times, 
            #                                                                 num_neighbors = num_neighbors, 
            #                                                                 time_gap = time_gap)
            
            # batch_neg_dst_node_embeddings = model[0].compute_node_embeddings(pe = final_trained_positional_encoding, 
            #                                                                 node_ids = batch_neg_dst_node_ids, 
            #                                                                 node_interact_times = batch_node_interact_times, 
            #                                                                 num_neighbors = num_neighbors, 
            #                                                                 time_gap = time_gap)

            # spatial_node_embeddings = model[0].linear_transformer(pe = final_trained_positional_encoding)

            batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
            batch_node_ids = np.array(list(set(batch_node_ids)))
            positional_encoding = final_trained_positional_encoding
            if positional_encoding.shape[1] > num_fft_batches:
                positional_encoding = torch.clone(positional_encoding[:,-num_fft_batches:, :])
                # new_positional_encoding = torch.clone(positional_encoding[-num_fft_batches:])
                # del positional_encoding
                # positional_encoding = new_positional_encoding

            # (batch_size, pe_dim)
            fft_current_positional_encoding = model[0].fourier_transform_pe(batch_node_ids, positional_encoding, batch_idx)
            current_positional_encoding = torch.clone(positional_encoding[:, -1, :])
            current_positional_encoding[torch.from_numpy(batch_node_ids)] = fft_current_positional_encoding

            spatial_node_embeddings = model[0].linear_transformer(pe = current_positional_encoding, 
                                                                    min_time = batch_node_interact_times.min(),
                                                                    num_neighbors = num_neighbors,
                                                                    time_gap = time_gap)
            
            batch_src_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                            node_ids = batch_src_node_ids, 
                                                                            node_interact_times = batch_node_interact_times, 
                                                                            num_neighbors = num_neighbors, 
                                                                            time_gap = time_gap)
            
            batch_dst_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                            node_ids = batch_dst_node_ids, 
                                                                            node_interact_times = batch_node_interact_times, 
                                                                            num_neighbors = num_neighbors, 
                                                                            time_gap = time_gap)
            
            batch_neg_src_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                            node_ids = batch_neg_src_node_ids, 
                                                                            node_interact_times = batch_node_interact_times, 
                                                                            num_neighbors = num_neighbors, 
                                                                            time_gap = time_gap)
            
            batch_neg_dst_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                            node_ids = batch_neg_dst_node_ids, 
                                                                            node_interact_times = batch_node_interact_times, 
                                                                            num_neighbors = num_neighbors, 
                                                                            time_gap = time_gap)

            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid().clamp(0, 1)
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().clamp(0, 1)

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
            # 

            ### update PE
            batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
            batch_node_ids = np.array(list(set(batch_node_ids)))
                
            # batch_current_positional_encoding = model[0].aggregated_pe(pe = current_positional_encoding, 
            #                                                         node_ids = batch_node_ids, 
            #                                                         batch_src_node_ids = batch_src_node_ids, 
            #                                                         batch_dst_node_ids = batch_dst_node_ids, 
            #                                                         )
            # current_positional_encoding[torch.from_numpy(batch_node_ids)] = batch_current_positional_encoding
            
            positional_encoding = torch.cat([positional_encoding, current_positional_encoding.unsqueeze(1)], dim = 1)
            ###

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics
