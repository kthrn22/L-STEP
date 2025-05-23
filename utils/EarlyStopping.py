import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, save_trained_pe: str = "no PE if not LSTEP", 
                 save_spatial_ne: str = "no PE if not LSTEP", model_name: str = None,
                 ):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        
        if model_name.startswith("LSTEP"):
            self.save_trained_positional_encoding_path = os.path.join(save_model_folder, f"{save_trained_pe}.pkl")
            self.save_spatial_node_embeddings_path = os.path.join(save_model_folder, f"{save_spatial_ne}.pkl")
        else:
            self.save_trained_positional_encoding_path = None
        
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module, final_trained_positional_encoding = None, spatial_node_embeddings = None):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            
            if self.model_name.startswith("LSTEP"):
                self.save_pe(final_trained_positional_encoding)
                self.save_ne(spatial_node_embeddings)
                
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_pe(self, final_trained_positional_encoding):
        if final_trained_positional_encoding is not None:
            self.logger.info(f"save trained positional encoding {self.save_trained_positional_encoding_path}")
            torch.save(final_trained_positional_encoding, self.save_trained_positional_encoding_path)

    def save_ne(self, spatial_node_embeddings):
        if spatial_node_embeddings is not None:
            self.logger.info(f"save spatial node embeddings {self.save_spatial_node_embeddings_path}")
            torch.save(spatial_node_embeddings, self.save_spatial_node_embeddings_path)

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)

    def load_pe(self):
        if self.logger is not None:
            self.logger.info(f"load positional encoding {self.save_trained_positional_encoding_path}")
        pe = torch.load(self.save_trained_positional_encoding_path)
        return pe
    
    def load_ne(self):
        if self.logger is not None:
            self.logger.info(f"load spatial node embeddings {self.save_spatial_node_embeddings_path}")
        ne = torch.load(self.save_spatial_node_embeddings_path)
        return ne

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        if self.logger is not None:
            self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
