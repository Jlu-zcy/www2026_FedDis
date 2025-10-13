import os
import yaml
import argparse
import torch
import numpy as np
from datetime import datetime
import pickle

from lib.utils import init_seed, load_graph, get_log_dir, get_project_path
from lib.dataloader import get_dataloader, STDataloader_T
from lib.logger import get_logger
from models.our_model import DisST
from train import Trainer

def get_sub_adj(adj_cut, nodes):
    sub_adj = adj_cut[np.ix_(nodes, nodes)]
    return torch.tensor(sub_adj, dtype=torch.float)

def test_federated_models(args):
    
    logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
    logger.info("Start testing federated learning models...")    
    project_path = get_project_path()
    args.graph_file = os.path.join(project_path, args.graph_file)
    args.data_dir = os.path.join(project_path, args.data_dir)    
    A = load_graph(args.graph_file, device=args.device)
    init_seed(args.seed)   
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )
    scaler = dataloader['scaler']   
    graph_partition_file = os.path.join(args.log_dir, 'graph_partition.pkl')
    if os.path.exists(graph_partition_file):
        logger.info("Load graph partition result...")
        with open(graph_partition_file, 'rb') as f:
            partition_data = pickle.load(f)
            parts = partition_data['parts']
            adj_cut = partition_data['adj_cut']
            clients_nodes = partition_data['clients_nodes']
            num_clients = partition_data['num_clients']
            
    logger.info(f'Client nodes: {[len(nodes) for nodes in clients_nodes]}')
    
    x_train = dataloader['train'].dataset.tensors[0].cpu().numpy()
    y_train = dataloader['train'].dataset.tensors[1].cpu().numpy()
    time_train = dataloader['train'].dataset.tensors[2].cpu().numpy()
    c_train = dataloader['train'].dataset.tensors[3].cpu().numpy()
    x_val = dataloader['val'].dataset.tensors[0].cpu().numpy()
    y_val = dataloader['val'].dataset.tensors[1].cpu().numpy()
    time_val = dataloader['val'].dataset.tensors[2].cpu().numpy()
    c_val = dataloader['val'].dataset.tensors[3].cpu().numpy()
    x_test = dataloader['test'].dataset.tensors[0].cpu().numpy()
    y_test = dataloader['test'].dataset.tensors[1].cpu().numpy()
    c_test = dataloader['test'].dataset.tensors[2].cpu().numpy()
    client_loaders = []
    val_loader_list = []
    test_loader_list = []    
    all_results = []
    
    for cid in range(num_clients):
        logger.info(f'Test client {cid} model...')       
        model_path = os.path.join(args.log_dir, f'client_{cid}_best_model_round.pth')           
        nodes = clients_nodes[cid]
        x_split = x_train[:, :, nodes, :]
        y_split = y_train[:, :, nodes, :]
        time_split = time_train[:]
        c_split = c_train[:, :, nodes, :]
        train_loader = STDataloader_T(
            x_split, y_split, time_split, c_split,
            args.batch_size, device=args.device, shuffle=True
        )
        client_loaders.append(train_loader)
        x_val_split = x_val[:, :, nodes, :]
        y_val_split = y_val[:, :, nodes, :]
        time_val_split = time_val[:]
        c_val_split = c_val[:, :, nodes, :]
        val_loader = STDataloader_T(
            x_val_split, y_val_split, time_val_split, c_val_split,
            args.batch_size, device=args.device, shuffle=False
        )
        val_loader_list.append(val_loader)
        x_test_split = x_test[:, :, nodes, :]
        y_test_split = y_test[:, :, nodes, :]
        c_test_split = c_test[:, :, nodes, :]
        test_loader = STDataloader_T(
            x_test_split, y_test_split, None, c_test_split,
            args.batch_size, device=args.device, shuffle=False, train_flag=False
        )
        test_loader_list.append(test_loader)       
        sub_adj = get_sub_adj(adj_cut, nodes).to(args.device)       
        args.num_nodes = len(nodes)
        model = DisST(args=args, adj=sub_adj, in_channels=args.d_input, embed_size=args.d_model,
                        T_dim=args.input_length, output_T_dim=12, output_dim=args.d_output, device=args.device).to(args.device)
                
        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])            
                       
        client_dataloader = {
            'train': client_loaders[cid],
            'val': val_loader_list[cid],
            'test': test_loader_list[cid],
            'scaler': scaler
        }        
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_init),
            dataloader=client_dataloader,
            graph=sub_adj,
            graph2=sub_adj,
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                torch.optim.Adam(model.parameters(), lr=args.lr_init),
                mode='min', factor=0.5, patience=args.lr_patience, threshold=0.0001,
                threshold_mode='rel', min_lr=0.000005, eps=1e-08),
            args=args,
            logger=logger
        )        
        test_results = trainer.test(
            model,
            test_loader,
            scaler,
            sub_adj,
            logger,
            args,
            cid=cid
        )        
        logger.info(f'Client {cid} test results: MAE={test_results[0,0]:.4f}, MAPE={test_results[0,1]*100:.4f}%, RMSE={test_results[0,2]:.4f}')
        all_results.append(test_results)
    
    if all_results:
        avg_results = np.mean(all_results, axis=0)
        logger.info(f'All clients average test results: MAE={avg_results[0,0]:.4f}, MAPE={avg_results[0,1]*100:.4f}%, RMSE={avg_results[0,2]:.4f}')
        
        summary_path = os.path.join(args.log_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Federated learning model test summary\n')
            f.write(f'Test time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Client number: {num_clients}\n\n')            
            for cid, result in enumerate(all_results):
                f.write(f'Client {cid}: MAE={result[0,0]:.4f}, MAPE={result[0,1]*100:.4f}%, RMSE={result[0,2]:.4f}\n')            
            f.write(f'\nAverage results: MAE={avg_results[0,0]:.4f}, MAPE={avg_results[0,1]*100:.4f}%, RMSE={avg_results[0,2]:.4f}\n')        
        logger.info(f'Test summary saved to: {summary_path}')
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to log directory containing trained models')
    args_cmd = parser.parse_args()
    
    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)
    
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    args = Struct(**config)
    args.log_dir = args_cmd.log_dir
    
    if not hasattr(args, 'ablation'):
        args.ablation = 'all'
    
    test_federated_models(args)

if __name__ == '__main__':
    main()
