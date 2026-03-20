import os
import yaml
import argparse
import torch
import numpy as np
import networkx as nx
import pymetis
from collections import defaultdict
import pickle

from lib.utils import (
    init_seed,
    load_graph,
    get_log_dir,
    get_project_path,
)
from lib.dataloader import get_dataloader, STDataloader_T
from lib.logger import get_logger
from models.our_model import DisST
from client import Client
from server import Server
from federated import FederatedRunner

def get_sub_adj(adj_cut, nodes):
    sub_adj = adj_cut[np.ix_(nodes, nodes)]
    return torch.tensor(sub_adj, dtype=torch.float)

def main(args):
    args.log_dir = get_log_dir(args)
    if not os.path.isdir(args.log_dir) and not args.debug:
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
    logger.info("Start federated learning training...")
    
    project_path = get_project_path()
    args.graph_file = os.path.join(project_path, args.graph_file)
    args.data_dir = os.path.join(project_path, args.data_dir)
    
    A = load_graph(args.graph_file, device=args.device)
    adj = load_graph(args.graph_file, return_numpy=True)
    init_seed(args.seed)
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )
    scaler = dataloader['scaler']
    num_clients = args.num_clients
    
    G = nx.from_numpy_array(adj)
    cuts, parts = pymetis.part_graph(num_clients, G)
    parts = list(parts)
    
    adj_cut = adj.copy()
    N = adj.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if parts[i] != parts[j] and adj[i, j] != 0:
                adj_cut[i, j] = 0
                adj_cut[j, i] = 0
    
    subgraph_nodes = defaultdict(list)
    for node, part in enumerate(parts):
        subgraph_nodes[part].append(node)
    clients_nodes = []
    for part in sorted(subgraph_nodes.keys()):
        clients_nodes.append(subgraph_nodes[part])
    
    graph_partition_file = os.path.join(args.log_dir, 'graph_partition.pkl')
    partition_data = {
        'parts': parts,
        'adj_cut': adj_cut,
        'num_clients': num_clients,
        'clients_nodes': clients_nodes,
        'subgraph_nodes': dict(subgraph_nodes)
    }
    with open(graph_partition_file, 'wb') as f:
        pickle.dump(partition_data, f)
    logger.info(f'Graph partition result has been saved to: {graph_partition_file}')
    logger.info(f'Client nodes: {clients_nodes}')
    
    
    x_train = dataloader['train'].dataset.tensors[0].cpu().numpy()
    y_train = dataloader['train'].dataset.tensors[1].cpu().numpy()  
    x_val = dataloader['val'].dataset.tensors[0].cpu().numpy()
    y_val = dataloader['val'].dataset.tensors[1].cpu().numpy()  
    x_test = dataloader['test'].dataset.tensors[0].cpu().numpy()
    y_test = dataloader['test'].dataset.tensors[1].cpu().numpy()  
    client_loaders = []
    val_loader_list = []
    test_loader_list = []


    for cid in range(num_clients):
        nodes = clients_nodes[cid]
        x_split = x_train[:, :, nodes, :]
        y_split = y_train[:, :, nodes, :]   
        train_loader = STDataloader_T(
            x_split, y_split, 
            args.batch_size, device=args.device, shuffle=True
        )
        client_loaders.append(train_loader)
        x_val_split = x_val[:, :, nodes, :]
        y_val_split = y_val[:, :, nodes, :]      
        val_loader = STDataloader_T(
            x_val_split, y_val_split, 
            args.batch_size, device=args.device, shuffle=False
        )
        val_loader_list.append(val_loader)
        x_test_split = x_test[:, :, nodes, :]
        y_test_split = y_test[:, :, nodes, :]      
        test_loader = STDataloader_T(
            x_test_split, y_test_split, 
            args.batch_size, device=args.device, shuffle=False, train_flag=False
        )
        test_loader_list.append(test_loader)
    
    server = Server(args=args, logger=logger)
    clients = []
    for cid in range(num_clients):
        nodes = clients_nodes[cid]
        sub_adj = get_sub_adj(adj_cut, nodes).to(args.device)
        args.num_nodes = len(nodes)
        local_model = DisST(args=args, adj=sub_adj, in_channels=args.d_input, embed_size=args.d_model,
                    T_dim=args.input_length, output_T_dim=12, output_dim=args.d_output, device=args.device).to(args.device)
        
        dataloader_dict = {
            'train': client_loaders[cid],
            'val': val_loader_list[cid],
            'test': test_loader_list[cid],
            'scaler': scaler
        }
        clients.append(Client(cid, local_model, dataloader_dict, args, logger, sub_adj))
    logger.info(f'Number of clients: {num_clients}')

    federated_runner = FederatedRunner(server, clients, args, logger, val_loader_list[0], test_loader_list[0], scaler, A)
    federated_runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args_cmd = parser.parse_args()
    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**config)
    main(args)
