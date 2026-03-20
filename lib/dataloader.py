import os
import torch
import numpy as np

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.std = np.where(self.std == 0, 1e-8, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor:            
            if type(self.mean) == np.ndarray:
                std_tensor = torch.from_numpy(self.std).to(data.device).type(data.dtype)
                mean_tensor = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
                return (data * std_tensor) + mean_tensor
            elif type(self.mean) == torch.Tensor:                
                return (data * self.std) + self.mean
            else:               
                std_tensor = torch.tensor(self.std, device=data.device, dtype=data.dtype)
                mean_tensor = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
                return (data * std_tensor) + mean_tensor
        else:            
            return (data * self.std) + self.mean

class MinMax01Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor:            
            if type(self.min) == np.ndarray:
                min_tensor = torch.from_numpy(self.min).to(data.device).type(data.dtype)
                max_tensor = torch.from_numpy(self.max).to(data.device).type(data.dtype)
                return (data * (max_tensor - min_tensor) + min_tensor)
            elif type(self.min) == torch.Tensor:                
                return (data * (self.max - self.min) + self.min)
            else:               
                min_tensor = torch.tensor(self.min, device=data.device, dtype=data.dtype)
                max_tensor = torch.tensor(self.max, device=data.device, dtype=data.dtype)
                return (data * (max_tensor - min_tensor) + min_tensor)
        else:            
            return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor:           
            if type(self.min) == np.ndarray:
                min_tensor = torch.from_numpy(self.min).to(data.device).type(data.dtype)
                max_tensor = torch.from_numpy(self.max).to(data.device).type(data.dtype)
                return ((data + 1.) / 2.) * (max_tensor - min_tensor) + min_tensor
            elif type(self.min) == torch.Tensor:                
                return ((data + 1.) / 2.) * (self.max - self.min) + self.min
            else:               
                min_tensor = torch.tensor(self.min, device=data.device, dtype=data.dtype)
                max_tensor = torch.tensor(self.max, device=data.device, dtype=data.dtype)
                return ((data + 1.) / 2.) * (max_tensor - min_tensor) + min_tensor
        else:
            
            return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def STDataloader_T(X, Y, batch_size, device,shuffle=True, drop_last=True,train_flag=True):

    TensorFloat = torch.FloatTensor
    TensorInt = torch.LongTensor
    if train_flag:
        X, Y = TensorFloat(X).to(device), TensorFloat(Y).to(device)
        data = torch.utils.data.TensorDataset(X, Y)
    else:
        X, Y = TensorFloat(X).to(device), TensorFloat(Y).to(device)
        data = torch.utils.data.TensorDataset(X, Y)

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    return scalar

def get_dataloader(data_dir, dataset, batch_size, test_batch_size, device, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        
    scaler = normalize_data(data['x_train'], scalar_type)
    
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category]) # y 其实不需要transform
   
    dataloader = {}
    dataloader['train'] = STDataloader_T(
        data['x_train'],
        data['y_train'],
        batch_size,
        device=device,
        shuffle=True
    )

    dataloader['val'] = STDataloader_T(
        data['x_val'], 
        data['y_val'],
        test_batch_size,
        device=device, 
        shuffle=False
    )
    dataloader['test'] = STDataloader_T(
        data['x_test'], 
        data['y_test'], 
        test_batch_size,
        device=device, 
        shuffle=False, 
        drop_last=False,
        train_flag=False
    )
    dataloader['scaler'] = scaler
    return dataloader
