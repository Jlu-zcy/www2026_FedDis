import os
import random
import torch
import numpy as np
from datetime import datetime


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def masked_mse_loss(mask_value):
    def loss(preds, labels):
        mse = mse_torch(pred=preds, true=labels, mask_value=mask_value)
        return mse
    return loss

def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def disp(x, name):
    print(f'{name} shape: {x.shape}')

def get_model_params(model_list):
    model_parameters = []
    for m in model_list:
        if m != None:
            model_parameters += list(m.parameters())
    return model_parameters

def get_log_dir(args):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
    return log_dir 

def load_graph(adj_file, device='cpu', return_numpy=False):
    graph = np.load(adj_file)['adj_mx']
    if return_numpy:
        return graph
    graph = torch.tensor(graph, device=device, dtype=torch.float)
    return graph


def get_project_path():
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return project_path

def find_last(search, target,start=0):
    loc = search.find(target,start)
    end_loc=loc
    while loc != -1:
        end_loc=loc
        start = loc+1
        loc = search.find(target,start)
    return end_loc

def print_model_parameters(model, logger=None, prefix=""):
    total_params = 0
    trainable_params = 0
    
    if logger:
        logger.info(f"{prefix}=== model parameters detailed information ===")
    else:
        print(f"{prefix}=== model parameters detailed information ===")
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        param_info = f"{prefix}parameter name: {name}, shape: {param.shape}, parameter count: {param_count}"
        if param.requires_grad:
            param_info += ", trainable"
        else:
            param_info += ", not trainable"
        
        if logger:
            logger.info(param_info)
        else:
            print(param_info)
    
    summary = f"{prefix}total parameters: {total_params}, trainable parameters: {trainable_params}"
    if logger:
        logger.info(summary)
    else:
        print(summary)
    
    return total_params, trainable_params

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def mae_torch_test(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    return torch.mean(torch.abs(true-pred))

def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def mse_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    return torch.mean(torch.square(true - pred))

def rmse_torch(pred, true, mask_value=None):
    return torch.sqrt(mse_torch(pred, true, mask_value))

def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))

def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def mse_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.square(pred - true))

def test_metrics(pred, true, mask1=5, mask2=5):
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
        mse  = mse_np(pred, true, mask1)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch_test(pred, true, mask1).item()
        mape = mape_torch(pred, true, mask2).item()
        rmse  = rmse_torch(pred, true, mask1).item()
    else:
        raise TypeError
    return mae, mape, rmse