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
import pdb


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    train_node_ids = np.array(list(set(train_data.src_node_ids.tolist() + train_data.dst_node_ids.tolist())))
    # train_node_ids = np.array(list(set(train_data.src_node_ids.tolist() + train_data.dst_node_ids.tolist())))
    # train_node_ids = np.array(list(set(train_data.src_node_ids.tolist() + train_data.dst_node_ids.tolist())))

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # pdb.set_trace()

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

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_trained_pe = f'{args.model_name}_pe_seed{args.seed}'
        
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        #### define dynamic backbone

        dynamic_backbone = TempGT(node_raw_features = node_raw_features,
                                  edge_raw_features = edge_raw_features,
                                  neighbor_sampler = train_neighbor_sampler,
                                  full_neighbor_sampler = full_neighbor_sampler,
                                  pe_dim = args.position_feat_dim,
                                  num_neighbors = args.num_neighbors,
                                  time_feat_dim = args.time_feat_dim,
                                  num_fft_batches = args.num_fft_batches,
                                  seq_len = node_raw_features.shape[0],
                                  num_heads = args.num_heads,
                                  transformer_depth = args.transformer_depth,
                                  num_layers = args.num_layers,
                                  dropout = args.dropout,
                                  transformer = "linformer", 
                                  device = args.device)
        

        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)

        if args.pre_trained == 'True':
            logger.info(f'Loading best model and continue training ...')
            model.load_state_dict(torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_seed{}.pkl".format(args.dataset_name, run, run), map_location = None))
            # final_trained_positional_encoding = torch.load("./saved_models/TempGT/{}/TempGT_seed{}/TempGT_pe_seed{}.pkl".format(args.dataset_name, run, run)).to(args.device)

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        if args.pre_trained == 'False':
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, save_trained_pe = args.save_trained_pe, 
                                       logger=logger, model_name=args.model_name)

        pe_loss_func = nn.MSELoss()
        loss_func = nn.BCELoss()

        # compute initial_positional_encoding 
        num_nodes = node_raw_features.shape[0]
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        for _, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]            

            edge_index = torch.from_numpy(np.array([batch_src_node_ids.tolist() + batch_dst_node_ids.tolist(),
                                                    batch_dst_node_ids.tolist() + batch_src_node_ids.tolist()]))
            break 

        # [N, pe_dim]
        k = min(num_nodes, args.position_feat_dim)
        initial_positional_encoding = LaplacianPE(edge_index, num_nodes, k)
        initial_positional_encoding = initial_positional_encoding.to(args.device)

        pdb.set_trace()

        ###
        node_embeddings = torch.zeros((node_raw_features.shape))
        ###

        for epoch in range(args.num_epochs):
            # initalize postional encoding
            # positional_encoding = torch.zeros((num_nodes, len(train_idx_data_loader), args.position_feat_dim)).to(args.device)

            positional_encoding = torch.tensor([])

            model.train()
            
            model[0].set_neighbor_sampler(train_neighbor_sampler)

            # torch.autograd.set_detect_anomaly(True)
            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            last_batch_idx = len(train_idx_data_loader_tqdm) - 1
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                positional_encoding = positional_encoding.to(args.device)
                train_data_indices = train_data_indices.numpy()
                
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                
                if batch_idx == 0:
                    current_positional_encoding = None
                else:
                    batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
                    batch_node_ids = np.array(list(set(batch_node_ids)))

                    ###* NEW WAY OF MAINTAINING positional_encoding, only apply FFT on the num_fft_batches
                    ### recent positional encodings to obtain the positional encoding for current batch
                    if positional_encoding.shape[1] > args.num_fft_batches:
                        positional_encoding = torch.clone(positional_encoding[:,-args.num_fft_batches:, :])
                        # print(positional_encoding[1].shape)
                        # del positional_encoding
                        # positional_encoding = new_positional_encoding
                    ###
                    ###

                    # (batch_size, pe_dim)
                    fft_current_positional_encoding = model[0].fourier_transform_pe(batch_node_ids, positional_encoding, batch_idx)
                    current_positional_encoding = torch.clone(positional_encoding[:, -1, :])
                    current_positional_encoding[torch.from_numpy(batch_node_ids)] = fft_current_positional_encoding


                # spatial_node_embeddings = model[0].linear_transformer(pe = current_positional_encoding)
                
                ###* NEW COMPONENT
                ### Apply graph transformer on temporal graph (with links accumulated from 1st batch to (i - 1)-th batch, 
                ### where i is the current batch we are processing) 
                ### to obtain node embeddings -> spatial_node_embeddings [num_nodes, node_feat_dim]
                spatial_node_embeddings = model[0].linear_transformer(pe = current_positional_encoding, 
                                                                    min_time = batch_node_interact_times.min(),
                                                                    num_neighbors = args.num_neighbors,
                                                                    time_gap = args.time_gap,
                                                                    train_node_ids = train_node_ids)
                ###
                ###

                # batch_src_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
                #                                                              node_ids = batch_src_node_ids, 
                #                                                              node_interact_times = batch_node_interact_times, 
                #                                                              num_neighbors = args.num_neighbors, 
                #                                                              time_gap = args.time_gap)
                

                # batch_dst_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
                #                                                              node_ids = batch_dst_node_ids, 
                #                                                              node_interact_times = batch_node_interact_times, 
                #                                                              num_neighbors = args.num_neighbors, 
                #                                                              time_gap = args.time_gap)
                
                # batch_neg_src_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
                #                                                              node_ids = batch_neg_src_node_ids, 
                #                                                              node_interact_times = batch_node_interact_times, 
                #                                                              num_neighbors = args.num_neighbors, 
                #                                                              time_gap = args.time_gap)

                # batch_neg_dst_node_embeddings = model[0].compute_node_embeddings(pe = current_positional_encoding, 
                #                                                              node_ids = batch_neg_dst_node_ids, 
                #                                                              node_interact_times = batch_node_interact_times, 
                #                                                              num_neighbors = args.num_neighbors, 
                #                                                              time_gap = args.time_gap)

        
                ###* Retrieve embedding learned by Graph Transformer (stored in spatial_node_embeddings) 
                ### Combine spatial_node_embeddings of u with the current temporal node embeddings of u at current time t
                ### to get the final representation for u
                batch_src_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                             node_ids = batch_src_node_ids, 
                                                                             node_interact_times = batch_node_interact_times, 
                                                                             num_neighbors = args.num_neighbors, 
                                                                             time_gap = args.time_gap)
                

                batch_dst_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings,
                                                                             node_ids = batch_dst_node_ids, 
                                                                             node_interact_times = batch_node_interact_times, 
                                                                             num_neighbors = args.num_neighbors, 
                                                                             time_gap = args.time_gap)
                
                batch_neg_src_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings, 
                                                                             node_ids = batch_neg_src_node_ids, 
                                                                             node_interact_times = batch_node_interact_times, 
                                                                             num_neighbors = args.num_neighbors, 
                                                                             time_gap = args.time_gap)

                batch_neg_dst_node_embeddings = model[0].compute_node_embeddings(spatial_node_embeddings = spatial_node_embeddings, 
                                                                             node_ids = batch_neg_dst_node_ids, 
                                                                             node_interact_times = batch_node_interact_times, 
                                                                             num_neighbors = args.num_neighbors, 
                                                                             time_gap = args.time_gap)
                
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid().clamp(0, 1)
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().clamp(0, 1)

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                lp_loss = loss_func(input=predicts, target=labels)

                # get similarity of positional encodings of positive and negative links and compute PE loss
                # (batch_size, pe_dim)
                if current_positional_encoding is not None:
                    pos_src_pe, pos_dst_pe = current_positional_encoding[torch.from_numpy(batch_src_node_ids)], \
                                            current_positional_encoding[torch.from_numpy(batch_dst_node_ids)]
                    neg_src_pe, neg_dst_pe = current_positional_encoding[torch.from_numpy(batch_neg_src_node_ids)], \
                                            current_positional_encoding[torch.from_numpy(batch_neg_dst_node_ids)]
                    positive_pe_loss = pe_loss_func(pos_src_pe, pos_dst_pe)
                    negative_pe_loss = pe_loss_func(neg_src_pe, neg_dst_pe)
                    pe_loss = positive_pe_loss - (args.neg_sample_weight) * negative_pe_loss
                else:
                    pe_loss = torch.tensor(0).to(torch.float32).to(args.device)

                train_losses.append(lp_loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                loss = (1 - args.pe_weight) * lp_loss + args.pe_weight * pe_loss
    
                # print("Updating Positional Encoding ...")
                if batch_idx == 0:
                    positional_encoding = torch.cat([positional_encoding, initial_positional_encoding.unsqueeze(1)], dim = 1)
                else:
                    # (batch_size, pe_dim)
                    # batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
                    # batch_node_ids = np.array(list(set(batch_node_ids)))
                    
                    # batch_current_positional_encoding = model[0].aggregated_pe(pe = current_positional_encoding, 
                    #                                                      node_ids = batch_node_ids, 
                    #                                                      batch_src_node_ids = batch_src_node_ids, 
                    #                                                      batch_dst_node_ids = batch_dst_node_ids, 
                    #                                                      )
                    # current_positional_encoding[torch.from_numpy(batch_node_ids)] = batch_current_positional_encoding
                    
                    positional_encoding = torch.cat([positional_encoding, current_positional_encoding.unsqueeze(1)], dim = 1)
                
                if current_positional_encoding is not None and batch_idx == last_batch_idx:
                    current_positional_encoding = current_positional_encoding.detach().cpu()
                # if current_positional_encoding is not None:
                #     current_positional_encoding = current_positional_encoding.detach().cpu()
                positional_encoding = positional_encoding.detach().cpu()
                # spatial_node_embeddings = spatial_node_embeddings.detach().cpu()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, lp loss: {lp_loss.item()}, pe loss: {pe_loss.item()}')

            # ####### evaluate
            # final_trained_positional_encoding = current_positional_encoding.to(args.device)
            final_trained_positional_encoding = positional_encoding.to(args.device)
            # spatial_node_embeddings = spatial_node_embeddings.to(args.device)

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                     spatial_node_embeddings = spatial_node_embeddings,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_fft_batches = args.num_fft_batches,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

          
            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                       spatial_node_embeddings = spatial_node_embeddings,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_fft_batches = args.num_fft_batches,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

           
            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           final_trained_positional_encoding = final_trained_positional_encoding,
                                                                           spatial_node_embeddings = spatial_node_embeddings,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_fft_batches = args.num_fft_batches,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                             spatial_node_embeddings = spatial_node_embeddings,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_fft_batches = args.num_fft_batches,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)

                
                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                
            for metric_name in new_node_val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]), True))
           
            early_stop = early_stopping.step(val_metric_indicator, model, final_trained_positional_encoding = final_trained_positional_encoding)

            if early_stop:
                break

        ######################### BEST MODEL
        # load the best model
        early_stopping.load_checkpoint(model)
        final_trained_positional_encoding = early_stopping.load_pe()

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                     spatial_node_embeddings = spatial_node_embeddings,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_fft_batches = args.num_fft_batches,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                       spatial_node_embeddings = spatial_node_embeddings,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_fft_batches = args.num_fft_batches,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   final_trained_positional_encoding = final_trained_positional_encoding,
                                                                   spatial_node_embeddings = spatial_node_embeddings,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_fft_batches = args.num_fft_batches,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                     spatial_node_embeddings = spatial_node_embeddings,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_fft_batches = args.num_fft_batches,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
