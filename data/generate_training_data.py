from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import json


def load_dat_data(data_dir, dataset_name):
    desc_file = os.path.join(data_dir, dataset_name, 'desc.json')
    with open(desc_file, 'r') as f:
        desc = json.load(f)
    
    shape = desc['shape']
    print(f"{dataset_name} shape: {shape}")  
    dat_file = os.path.join(data_dir, dataset_name, 'data.dat')
    data = np.memmap(dat_file, mode='r', dtype=np.float32, shape=tuple(shape)).copy()    
    num_timesteps = data.shape[0]
    time_index = pd.date_range(
        start='2018-01-01 00:00:00', 
        periods=num_timesteps, 
        freq='5T'  
    )   
    print(f"data shape: {data.shape}")
    
    return data, time_index


def dat_to_dataframe(data, time_index):
    
    df_data = data[:, :, 0]  
    columns = [f'node_{i}' for i in range(data.shape[1])]
    df = pd.DataFrame(df_data, index=time_index, columns=columns)
    
    print(f"DataFrame: {df.shape}")
    print(f"DataFrame: {len(df.columns)}")
    
    return df


def generate_train_val_test(args):
    if args.data_source == 'hdf5':
        df = pd.read_hdf(args.traffic_df_filename)
        
    elif args.data_source == 'dat':
        data, time_index = load_dat_data(args.data_dir, args.dataset_name)
        df = dat_to_dataframe(data, time_index)
        
    else:
        raise ValueError(f"Unsupported Data Source Type: {args.data_source}")
    
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, 13, 1))
    
    num_samples = df.shape[0] - abs(x_offsets[0]) - abs(y_offsets[-1])
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    train_start = abs(x_offsets[0])
    train_end = train_start + num_train
    train_df = df.iloc[train_start:train_end]
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    return x, y


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/METR-LA", help="Output directory."
    )
    parser.add_argument(
        "--data_source", 
        type=str, 
        choices=['hdf5', 'dat'], 
        default='dat',
        help="Data source type: 'hdf5' for HDF5 files, 'dat' for .dat files."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/METR-LA/data.dat",
        help="Raw traffic readings (for HDF5 data source).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory (for .dat data source).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="METR-LA",
        help="Dataset name (for .dat data source).",
    )
    args = parser.parse_args()
    main(args)
