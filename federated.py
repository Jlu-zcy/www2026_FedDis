import copy
import os
import torch
import numpy as np

from lib.utils import print_model_parameters
from lib.logger import get_logger
from train import Trainer

class FederatedRunner:
    def __init__(self, server, clients, args, logger, val_loader, test_loader, scaler, graph):
        self.server = server
        self.clients = clients
        self.args = args
        self.logger = logger
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.graph = graph
        self.best_mae_per_client = [float('inf')] * len(self.clients)
        self.best_model_state_per_client = [None] * len(self.clients)
        self.best_round_per_client = [0] * len(self.clients)
        
        self.federated_lr_scheduler = None
        self.current_lr = self.args.lr_init
        self._init_federated_lr_scheduler()

    def _init_federated_lr_scheduler(self):
        import torch
        dummy_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=self.current_lr)
        self.federated_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            dummy_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.args.lr_patience, 
            threshold=0.0001, 
            threshold_mode='rel', 
            min_lr=0.000005, 
            eps=1e-08
        )
        self.logger.info(f"Federated learning rate scheduler has been initialized, initial learning rate: {self.current_lr}")

    def _update_federated_lr(self, round_val_loss):
        old_lr = self.current_lr
        self.federated_lr_scheduler.step(round_val_loss)
        self.current_lr = self.federated_lr_scheduler.optimizer.param_groups[0]['lr']
        
        if self.current_lr != old_lr:
            self.logger.info(f"Federated learning rate has been adjusted: {old_lr:.6f} -> {self.current_lr:.6f}")
        else:
            self.logger.info(f"Federated learning rate remains unchanged: {self.current_lr:.6f}")

    def run(self):
        import torch
        import numpy as np
        
        best_mae = float('inf')
        best_client_id = 0
        
        for rnd in range(1, self.args.num_rounds + 1):
            self.logger.info(f'===== Federated Round {rnd} =====')
            self.logger.info(f'current lr: {self.current_lr:.6f}')
            aggregation_config = self.server.get_aggregation_config()                        
            client_models = []
            client_train_infos = []
            client_embeddings = []  

            for client in self.clients:
                client.current_lr = self.current_lr
                trained_model = client.local_train(self.args.local_epochs)
                client_models.append(trained_model)
                client_train_infos.append(client.last_train_info)              
                graph_embedding, attention_weights = client.get_graph_embedding()
                client_embeddings.append(graph_embedding)
                       
            self.logger.info(f'-------- Round {rnd} all clients training completed ! --------')
            
            client_params_list = []
            for client in self.clients:
                if aggregation_config['aggregate_all']:
                    params = client.get_model_params()
                    for exclude_param in aggregation_config['exclude_params']:
                        if exclude_param in params:
                            del params[exclude_param]
                else:
                    include_params = aggregation_config['include_params'] or []
                    params = client.get_partial_params(include_params)
                client_params_list.append(params)
                       
            if len(client_embeddings) > 0 and hasattr(self.server, 'personalized_aggregate_params'):
                self.logger.info(f'Round {rnd} start personalized parameter aggregation...')
                    
                
                personalized_params_list = self.server.personalized_aggregate_params(
                    client_params_list, 
                    client_embeddings, 
                    param_names_to_aggregate=aggregation_config.get('include_params', None)
                )
                               
                for i, (client, personalized_params) in enumerate(zip(self.clients, personalized_params_list)):
                    client.update_model_params(personalized_params, update_strategy='partial')
                    self.logger.info(f'Client {client.cid} personalized parameters updated!')
                                    
            round_val_loss = sum(train_info['current_val_loss'] for train_info in client_train_infos) / len(client_train_infos)
            self.logger.info(f'Round {rnd} average val loss: {round_val_loss:.4f}')            
            self._update_federated_lr(round_val_loss)
                       
            for cid, client in enumerate(self.clients):                
                current_mae = client.last_train_info['best_loss']  
                self.logger.info(f'Client {cid} current MAE: {current_mae:.4f}')
                                
                if current_mae < self.best_mae_per_client[cid]:
                    self.best_mae_per_client[cid] = current_mae
                    self.best_model_state_per_client[cid] = copy.deepcopy(client.model.state_dict())
                    self.best_round_per_client[cid] = rnd
                    
                    if hasattr(self.args, 'log_dir') and self.args.log_dir:
                        import os
                        client_best_path = os.path.join(self.args.log_dir, f'client_{cid}_best_model_round.pth')
                        torch.save({
                            "round": rnd,
                            "model": self.best_model_state_per_client[cid],
                            "mae": current_mae,
                        }, client_best_path)
                        self.logger.info(f'Client {cid} best federation model has been saved to: {client_best_path} (MAE: {current_mae:.4f})')
                else:
                    self.logger.info(f'Client {cid} not improved, keeping the best history MAE: {self.best_mae_per_client[cid]:.4f}')
            
            best_client_this_round = min(enumerate(client_train_infos), key=lambda x: x[1]['best_loss'])
            best_client_id_this_round, best_train_info = best_client_this_round
            
            if best_train_info['best_loss'] < best_mae:
                best_mae = best_train_info['best_loss']
                best_client_id = best_client_id_this_round
                self.logger.info(f'############## The best client update is Client {best_client_id} (MAE: {best_mae:.4f}) #################')

def local_train_with_attention(trainer, local_epochs, args, cid):
    trainer.logger.info(f"Client {cid} begin attention training...")
    best_mae = float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    not_improved_count = 0
    best_model_state = None
    best_model_updated = False

    use_attention = (hasattr(trainer, 'client') and 
                    hasattr(trainer.client, 'aggregation_method') and 
                    trainer.client.aggregation_method == 'attention')

    for epoch in range(1, local_epochs + 1):
        train_epoch_loss = trainer.train_epoch(epoch, cid)
        if train_epoch_loss > 1e6:
            trainer.logger.warning('Gradient explosion detected. Ending...')
            break

        val_dataloader = trainer.val_loader if trainer.val_loader is not None else trainer.test_loader
        val_epoch_loss, current_mae = trainer.val_epoch(epoch, val_dataloader, cid)

        if current_mae < best_mae:
            best_mae = current_mae
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            not_improved_count = 0
            best_model_updated = True
            best_model_state = {
                "epoch": epoch,
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "mae": current_mae,
                "val_loss": val_epoch_loss,
            }          
            if use_attention:
                best_model_state["graph_aggregator"] = trainer.client.graph_aggregator.state_dict()            
        else:
            not_improved_count += 1

        if getattr(args, 'early_stop', False) and not_improved_count == args.early_stop_patience:
            trainer.logger.info(f"Validation performance didn't improve for {args.early_stop_patience} epochs. Training stops.")
            break
    
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state['model'])
        if use_attention and 'graph_aggregator' in best_model_state:
            trainer.client.graph_aggregator.load_state_dict(best_model_state['graph_aggregator'])
        trainer.logger.info(f'Client {cid} attention finish, best MAE: {best_mae:.4f}')

    return trainer.model, {
        'best_loss': best_mae,
        'best_val_loss': best_val_loss,
        'current_val_loss': val_epoch_loss,  
        'best_epoch': best_epoch,
        'best_model_updated': best_model_updated,
        'final_epoch': epoch if "epoch" in locals() else local_epochs
    }