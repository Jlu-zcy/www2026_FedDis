import os
import sys

import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
import numpy as np

from lib.utils import test_metrics, get_model_params, get_log_dir
from lib.logger import get_logger, PD_Stats
from models.our_model import DisST


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, lr_scheduler,args, graph2=None,load_state=None, logger=None, client=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.lr_scheduler=lr_scheduler
        self.args = args
        self.client = client
        if graph2 != None:
            self.test_graph=graph2
        else:
            self.test_graph=graph

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        
        if logger is not None:
            self.logger = logger
        else:
            args.log_dir = get_log_dir(args)
            if os.path.isdir(args.log_dir) == False and not args.debug:
                os.makedirs(args.log_dir, exist_ok=True)
            self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )

    def train_epoch(self, epoch, cid=None):
        
        self.model.train()
        
        if self.client and hasattr(self.client, 'graph_aggregator'):
            self.client.graph_aggregator.train()
            
        total_loss = 0
        total_lp = 0
        attentions = []
        mi_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train] Client {cid}', 
                   leave=True, ncols=100, position=1)
        
        for batch_idx, (data, target) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            D, S = self.model(data)
            
            attention = 0
            if self.client and hasattr(self.client, 'aggregation_method') and self.client.aggregation_method == 'attention':               
                node_embeddings = self.client._extract_node_embeddings()
                if node_embeddings is not None:
                    graph_embedding, attention_weights = self.client.graph_aggregator(node_embeddings)                       
                    attention_sparsity = torch.mean(attention_weights ** 2)
                    attention_entropy = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1))
                    attention = attention_sparsity + attention_entropy
                    attentions.append(attention.item())
         
            loss, _, lm, lp = self.model.calculate_loss(
                D, S, target, self.scaler, training=True
            )            
            total_loss_with_attention = loss + attention            
            total_loss_with_attention.backward()
            
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]), self.args.max_grad_norm)
                if self.client and hasattr(self.client, 'graph_aggregator'):
                    torch.nn.utils.clip_grad_norm_(
                        self.client.graph_aggregator.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()
            total_lp += lp.item()
            
            if type(lm) == int:
                mi_losses.append(lm)
            else:
                mi_losses.append(lm.item())
            
            postfix_info = {
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            }
            
            pbar.set_postfix(postfix_info)
        
        train_epoch_loss = total_loss / self.train_per_epoch
        avg_attention = np.mean(attentions) if attentions else 0        
        avg_total_loss = train_epoch_loss + avg_attention             
        self.logger.info(f'Train Epoch: Loss: {avg_total_loss:.6f}')
        
        return train_epoch_loss

    def val_epoch(self, epoch, val_dataloader, cid=None):
        
        self.model.eval()
        
        if self.client and hasattr(self.client, 'graph_aggregator'):
            self.client.graph_aggregator.eval()
        
        total_val_loss = 0
        total_lp = 0
        mi_losses = []
        
        pbar = tqdm(val_dataloader, desc=f'Epoch {epoch} [Val] Client {cid}', 
                   leave=True, ncols=100, position=2)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                D, S = self.model(data)
                
                loss, _, lm, lp = self.model.calculate_loss(
                    D, S, target, self.scaler, training=False
                )
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                if not torch.isnan(lp):
                    total_lp += lp.item()
                
                if type(lm) == int:
                    mi_losses.append(lm)
                else:
                    mi_losses.append(lm.item())
                
                postfix_info = {
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{total_val_loss/(batch_idx+1):.4f}',
                    'MAE': f'{total_lp/(batch_idx+1):.4f}'
                }
                
                pbar.set_postfix(postfix_info)
        
        val_loss = total_val_loss / len(val_dataloader)
        avg_lp = total_lp / len(val_dataloader)
        self.logger.info(f'Val Epoch: Loss: {val_loss:.6f}')
        
        return val_loss, avg_lp


    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args, cid=None):
        model.eval()
        y_pred = []
        y_true = []
        
        test_pbar = tqdm(dataloader, desc='Testing', ncols=100, position=0, leave=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_pbar):
                D, S  = model(data,graph)
                pred_output, att, D_hat, S_hat = model.predict_test(D, S)
                pred_output = pred_output.squeeze(1)
                target = target.squeeze(1)
                y_true.append(target)
                y_pred.append(pred_output)
                
                test_pbar.set_postfix({
                    'Batch': f'{batch_idx+1}/{len(dataloader)}',
                    'Processed': f'{len(y_pred)} samples'
                })
        
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))        
        test_results = []
        
        mae, mape, rmse = test_metrics(y_pred, y_true)
        logger.info("MAE: {:.2f}, MAPE: {:.4f}%, RMSE: {:.2f}".format(mae, mape * 100, rmse))
        test_results.append([mae, mape, rmse])

        return np.stack(test_results, axis=0)