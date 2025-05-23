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


from models.LSTEP import LSTEP
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler

from evaluate_model_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, load_link_prediction_best_configs_LSTEP
from utils.PositionalEncoding import LaplacianPE, RandomWalkPE


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    if args.load_best_configs:
        args = load_link_prediction_best_configs_LSTEP(args)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    train_node_ids = np.array(list(set(train_data.src_node_ids.tolist() + train_data.dst_node_ids.tolist())))
    
    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

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

    if args.ablation == "time_gap":
        args.ablation += str(args.time_gap)

    if args.ablation == "num_neighbors":
        args.ablation += str(args.num_neighbors)

    if args.ablation == "num_fft_batches":
        args.ablation += str(args.num_fft_batches)
    
    if args.ablation == "pe_weight":
        args.ablation += str(args.pe_weight)
    
    if args.ablation == "neg_sample_weight":
        args.ablation += str(args.neg_sample_weight)
    
    for run in range(args.num_runs):

        set_random_seed(seed=run)

        if run < args.start_seed or run > args.end_seed:
            continue

        args.seed = run
        args.save_model_name = f'{args.model_name + args.ablation}_seed{args.seed}'
        args.save_trained_pe = f'{args.model_name + args.ablation}_pe_seed{args.seed}'
        args.save_spatial_ne = f'{args.model_name + args.ablation}_ne_seed{args.seed}'
        
        all_scores_val = {'ap': [],
                      'roc': [],
                      'new_ap': [],
                      'new_roc': []}
        
        all_scores_test = {'ap': [],
                      'roc': [],
                      'new_ap': [],
                      'new_roc': []}

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name + args.ablation}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name + args.ablation}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
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

        use_weighted_sum = True if args.ablation == "weighted_sum" else False
        dynamic_backbone = LSTEP(node_raw_features = node_raw_features,
                                edge_raw_features = edge_raw_features,
                                neighbor_sampler = train_neighbor_sampler,
                                full_neighbor_sampler = full_neighbor_sampler,
                                pe_dim = args.position_feat_dim,
                                num_neighbors = args.num_neighbors,
                                time_feat_dim = args.time_feat_dim,
                                num_fft_batches = args.num_fft_batches,
                                dropout = args.dropout,
                                weighted_sum = use_weighted_sum,
                                concat_pe = args.concat_pe, 
                                device = args.device)

        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name +args.ablation}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name + args.ablation}/{args.dataset_name}/{args.save_model_name}/"
        if args.pre_trained == 'False':
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, save_trained_pe = args.save_trained_pe, 
                                       save_spatial_ne = args.save_spatial_ne,
                                       logger=logger, model_name=args.model_name)

        pe_loss_func = nn.MSELoss()
        loss_func = nn.BCELoss()
        weird_func = nn.MSELoss()
        # compute initial_positional_encoding 
        num_nodes = node_raw_features.shape[0]
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        for _, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]            

            _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

            edge_index = torch.from_numpy(np.array([batch_src_node_ids.tolist() + batch_dst_node_ids.tolist(),
                                                    batch_dst_node_ids.tolist() + batch_src_node_ids.tolist()]))
            
            break 

        # [N, pe_dsim]
        if args.model_name == "LSTEP_RWPE":
            initial_positional_encoding = RandomWalkPE(edge_index, num_nodes, args.position_feat_dim)
        else:
            k = min(num_nodes, args.position_feat_dim)
            initial_positional_encoding, edge_weight = LaplacianPE(edge_index, num_nodes, k)
        initial_positional_encoding = initial_positional_encoding.to(args.device)
        
        # pdb.set_trace()

        for epoch in range(args.num_epochs):
            
            model.train()

            positional_encoding = torch.tensor([])        
            model[0].set_neighbor_sampler(train_neighbor_sampler)

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
                    batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist() + batch_neg_dst_node_ids.tolist()
                    batch_node_ids = torch.from_numpy(np.array(batch_node_ids)).unique().numpy()

                    if positional_encoding.shape[1] > args.num_fft_batches:
                        positional_encoding = torch.clone(positional_encoding[:,-args.num_fft_batches:, :])
                        
                    # (batch_size, pe_dim)
                    fft_current_positional_encoding = model[0].fourier_transform_pe(batch_node_ids, positional_encoding, batch_idx)
                    current_positional_encoding = torch.clone(positional_encoding[:, -1, :])
                    current_positional_encoding[torch.from_numpy(batch_node_ids)] = fft_current_positional_encoding

                if current_positional_encoding is not None:
                    pos_src = model[0].combining_pe_raw_feat(pe = current_positional_encoding,
                                                            node_ids = batch_src_node_ids, 
                                                            node_interact_times = batch_node_interact_times, 
                                                            num_neighbors = args.num_neighbors,
                                                            time_gap = args.time_gap)

                    pos_dst = model[0].combining_pe_raw_feat(pe = current_positional_encoding,
                                                            node_ids = batch_dst_node_ids,
                                                            node_interact_times = batch_node_interact_times,
                                                            num_neighbors = args.num_neighbors,
                                                            time_gap = args.time_gap)
                    
                    neg_src = pos_src

                    neg_dst = model[0].combining_pe_raw_feat(pe = current_positional_encoding,
                                                            node_ids = batch_neg_dst_node_ids,
                                                            node_interact_times = batch_node_interact_times,
                                                            num_neighbors = args.num_neighbors,
                                                            time_gap = args.time_gap)    
                    

                    positive_probabilities = model[1](input_1=pos_src, input_2=pos_dst).squeeze(dim=-1).sigmoid().clamp(0, 1)
                    negative_probabilities = model[1](input_1=neg_src, input_2=neg_dst).squeeze(dim=-1).sigmoid().clamp(0, 1)

                    pos_src_pe, pos_dst_pe = current_positional_encoding[torch.from_numpy(batch_src_node_ids)], \
                                            current_positional_encoding[torch.from_numpy(batch_dst_node_ids)]
                    neg_src_pe, neg_dst_pe = current_positional_encoding[torch.from_numpy(batch_neg_src_node_ids)], \
                                            current_positional_encoding[torch.from_numpy(batch_neg_dst_node_ids)]
                    
                    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                    lp_loss = loss_func(input=predicts, target=labels)

                    positive_pe_loss = pe_loss_func(pos_src_pe, pos_dst_pe)
                    negative_pe_loss = pe_loss_func(neg_src_pe, neg_dst_pe)
                
                    pe_loss = positive_pe_loss - (args.neg_sample_weight) * negative_pe_loss

                    train_losses.append(lp_loss.item())
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                    loss = (1 - args.pe_weight) * lp_loss + args.pe_weight * pe_loss
                    
                else:
                    pe_loss = torch.tensor(0).to(torch.float32).to(args.device)

                if batch_idx == 0:
                    current_positional_encoding = initial_positional_encoding

                batch_node_ids = batch_src_node_ids.tolist() + batch_dst_node_ids.tolist()
                batch_node_ids = torch.from_numpy(np.array(batch_node_ids)).unique().numpy()
                
                new_node_pe = model[0].update_pe(
                        pe = current_positional_encoding, 
                        node_ids = batch_node_ids, 
                        edge_ids = batch_edge_ids, 
                        batch_src_node_ids = batch_src_node_ids, 
                        batch_dst_node_ids = batch_dst_node_ids, 
                        node_interact_times = batch_node_interact_times, 
                        current_time = batch_node_interact_times.max(),
                        num_neighbors = args.num_neighbors,
                        time_gap = args.time_gap)  

                if batch_idx > 0:
                    current_positional_encoding = new_node_pe
                    # current_positional_encoding[torch.from_numpy(batch_node_ids)] = new_node_pe
                                    
                positional_encoding = torch.cat([positional_encoding, current_positional_encoding.unsqueeze(1)], dim = 1)
                
                if current_positional_encoding is not None and batch_idx == last_batch_idx:
                    current_positional_encoding = current_positional_encoding.detach().cpu()
                
                positional_encoding = positional_encoding.detach().cpu()
                
                if batch_idx > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, pe loss: {pe_loss.item()}')

            # ####### evaluate
            # final_trained_positional_encoding = current_positional_encoding.to(args.device)
            # evaluate the best model
            final_trained_positional_encoding = positional_encoding.to(args.device)
            # early_stopping.load_checkpoint(model)
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_fft_batches = args.num_fft_batches,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,)

          
            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_fft_batches = args.num_fft_batches,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap,)

            val_ap = np.mean([val_metric['average_precision'] for val_metric in val_metrics])
            val_roc = np.mean([val_metric['roc_auc'] for val_metric in val_metrics])
                              
            new_val_ap = np.mean([val_metric['average_precision'] for val_metric in new_node_val_metrics])
            new_val_roc = np.mean([val_metric['roc_auc'] for val_metric in new_node_val_metrics])

            all_scores_val['ap'].append(val_ap)
            all_scores_val['roc'].append(val_roc)
            all_scores_val['new_ap'].append(new_val_ap)
            all_scores_val['new_roc'].append(new_val_roc)
               
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
                # pdb.set_trace()
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           final_trained_positional_encoding = final_trained_positional_encoding,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_fft_batches = args.num_fft_batches,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap,)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_fft_batches = args.num_fft_batches,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap,)
                                                                                            #  testing = True)

                test_ap = np.mean([test_metric['average_precision'] for test_metric in test_metrics])
                test_roc = np.mean([test_metric['roc_auc'] for test_metric in val_metrics])
                                
                new_test_ap = np.mean([test_metric['average_precision'] for test_metric in new_node_test_metrics])
                new_test_roc = np.mean([test_metric['roc_auc'] for test_metric in new_node_test_metrics])
                
                all_scores_test['ap'].append(test_ap)
                all_scores_test['roc'].append(test_roc)
                all_scores_test['new_ap'].append(new_test_ap)
                all_scores_test['new_roc'].append(new_test_roc)

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
        
        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model,
                                                                    final_trained_positional_encoding = final_trained_positional_encoding,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    evaluate_data=val_data,
                                                                    loss_func=loss_func,
                                                                    num_fft_batches = args.num_fft_batches,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap,)

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                    evaluate_data=new_node_val_data,
                                                                                    loss_func=loss_func,
                                                                                    num_fft_batches = args.num_fft_batches,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap,)

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   final_trained_positional_encoding = final_trained_positional_encoding,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_fft_batches = args.num_fft_batches,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap,)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     final_trained_positional_encoding = final_trained_positional_encoding,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_fft_batches = args.num_fft_batches,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap,)
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

        save_result_folder = f"./saved_results/{args.model_name + args.ablation}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name + args.ablation}.json")

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
