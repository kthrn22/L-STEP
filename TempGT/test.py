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

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
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

if __name__ == "__main__":
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

    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    dynamic_backbone = TempGT(node_raw_features = node_raw_features,
                                  edge_raw_features = edge_raw_features,
                                  neighbor_sampler = train_neighbor_sampler,
                                  pe_dim = args.position_feat_dim,
                                  num_neighbors = args.num_neighbors,
                                  time_feat_dim = args.time_feat_dim,
                                  num_batches = len(train_idx_data_loader),
                                  seq_len = args.batch_size,
                                  num_heads = args.num_heads,
                                  transformer_depth = args.transformer_depth,
                                  num_edge_layers = 2,
                                  dropout = args.dropout,
                                  transformer = "linformer", 
                                  device = args.device)
        
        # define positional encoding and transformer model

    link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
    model = nn.Sequential(dynamic_backbone, link_predictor)
    model = convert_to_gpu(model, device=args.device)
    loss_func = nn.BCELoss()

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    for run in range(args.num_runs):
        model.load_state_dict(torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_seed{}.pkl".format(args.dataset_name, run, run), map_location = None))
        final_trained_positional_encoding = torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_pe_seed{}.pkl".format(args.dataset_name, run, run)).to(args.device)

        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                    evaluate_data=new_node_val_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap)
        
        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   final_trained_positional_encoding = final_trained_positional_encoding,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        
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

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)
    
    for metric_name in test_metric_all_runs[0].keys():
        # logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        print(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        # logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
        #             f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        print(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        # logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        print(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        # logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
        #             f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')
        print(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')
