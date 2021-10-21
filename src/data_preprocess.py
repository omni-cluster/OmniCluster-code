'''
从原始文件夹读取txt文件，转换为npy格式
根据window_size,step_size对数据进行切割
根据clean_list选择数据
去除异常值并进行填充
平滑处理
归一化处理
'''
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import configs

CONFIG_PATH = 'configs'

def read_file(file):
    data = np.genfromtxt(file, dtype=np.float32, delimiter=",", skip_header=1)
    data = np.transpose(data)
    return data
def read_data_to_npy(dataset, dataset_path):
    """
    为了和原有读取数据顺序一致
    Args:
        dataset:
        dataset_path:

    Returns:num * index * time

    """
    data = []
    if dataset == "OMI":
        train_data_path = pathlib.Path(dataset_path).joinpath('train_data')
        test_data_path = pathlib.Path(dataset_path).joinpath('test_data')
        train_file_list = sorted(list(train_data_path.rglob("*.txt")))
        test_file_list = sorted(list(test_data_path.rglob("*.txt")))
        file_list = train_file_list + test_file_list
        for file in file_list:
            data.append(read_file(file))
    return data

def split_data(in_data, window_size, step_size):
    data = []
    for i_data in in_data:
        for i in range(0, i_data.shape[1], step_size):
            if i + window_size < i_data.shape[1] - 1:
                data.append(i_data[:, i: i + window_size])
    return np.array(data)


def choose_clean_data(in_data, clean_index_file):
    index = np.load(clean_index_file)
    return in_data[index]


def remove_extreme_data(in_data, top_n, mode):
    for data in in_data:
        for n_index_data in data:
            if mode == 'extreme':
                # 去除最大最小极值
                half_length = int(len(n_index_data) * top_n / 2)
                sorted_args = np.argsort(n_index_data)
                n_index_data[sorted_args[:half_length]] = np.nan
                n_index_data[sorted_args[-half_length:]] = np.nan
            elif mode == 'deviation-mean':
                # 去除偏离均值最大的5%的数据
                length = int(len(n_index_data) * top_n)
                mean = np.mean(n_index_data)
                abs_distance_mean = np.abs(n_index_data-mean)
                sorted_args = np.argsort(abs_distance_mean)
                n_index_data[sorted_args[-length:]] = np.nan
    data = deal_nan(in_data)
    return data


def deal_nan(in_data):
    for data in in_data:
        x = np.arange(data.shape[1])
        for n_index_data in data:
            if sum(np.isnan(n_index_data)) == n_index_data.shape:
                np.nan_to_num(n_index_data, copy=False)
            else:
                nan_index = np.where(np.isnan(n_index_data))[0]
                non_nan_index_x = np.setdiff1d(x, nan_index)
                non_nan_index_y = n_index_data[non_nan_index_x]
                n_index_data[nan_index] = np.interp(nan_index, non_nan_index_x, non_nan_index_y)
    return in_data


def moving_average(in_data, window=12, min_periods=1):
    for data in in_data:
        for i, n_index_data in enumerate(data):
            df = pd.Series(n_index_data)
            moving_avg = df.rolling(window=window, min_periods=min_periods).mean()
            moving_avg = moving_avg.values.flatten()
            data[i] = moving_avg
    return in_data


def feature_scaling(in_data, mode):
    sca = ""
    if len(in_data.shape) != 3:
        raise ValueError("Data must be a 3-D array")
    if mode == "norm":
        sca = MinMaxScaler()
    elif mode == "stand":
        sca = StandardScaler()
    for i, data in enumerate(in_data):
        data = data.transpose()
        data = sca.fit_transform(data)
        in_data[i] = data.transpose()
    return in_data


def main(paras):
    if paras["to_deal_dataset"] not in paras["all_dataset"]:
        raise ValueError("UNKNOWN DATASET " + str(paras["to_deal_dataset"]))
    data = read_data_to_npy(paras["to_deal_dataset"], paras["dataset_path"])
    if paras["if_split"]:
        data = split_data(data, paras["split_window_size"], paras["split_stride"])
    else:
        data = np.array(data)
    if paras["if_to_clean"]:
        data = choose_clean_data(data, paras["clean_file"])
    if paras["if_remove_extreme"]:
        data = remove_extreme_data(data, paras["extreme_per"], paras["remove_extreme_mode"])
    if paras["if_moving_average"]:
        data = moving_average(data, paras["moving_window_size"], paras["min_periods"])
    if paras["if_feature_scaling"]:
        data = feature_scaling(data, paras["mode"])
    np.save(paras["save_data_path"], data)


if __name__ == "__main__":
    config = configs.config(CONFIG_PATH)
    paras = {"to_deal_dataset": config.get('preprocess', 'to_deal_dataset'),
             "dataset_path": config.get('preprocess', 'dataset_path'),
             "all_dataset": json.loads(config.get('preprocess', 'all_dataset')),
             "save_data_path": pathlib.Path(config.get('preprocess', 'save_data_path')),
             "if_split": config.getboolean('preprocess', 'if_split'),
             "split_window_size": config.getint('preprocess', 'split_window_size'),
             "split_stride": config.getint('preprocess', 'split_stride'),
             "if_to_clean": config.getboolean('preprocess', 'if_to_clean'),
             "clean_file": config.get('preprocess', 'clean_file'),
             "if_remove_extreme": config.getboolean('preprocess', 'if_remove_extreme'),
             "extreme_per": config.getfloat('preprocess', 'extreme_per'),
             "remove_extreme_mode": config.get('preprocess', 'remove_extreme_mode'),
             "if_moving_average": config.getboolean('preprocess', 'if_moving_average'),
             "moving_window_size": config.getint('preprocess', 'moving_window_size'),
             "min_periods": config.getint('preprocess', 'min_periods'),
             "if_feature_scaling": config.getboolean('preprocess', 'if_feature_scaling'),
             "mode": config.get('preprocess', 'mode')}
    paras["save_data_path"].parent.mkdir(parents=True, exist_ok=True)
    main(paras)
