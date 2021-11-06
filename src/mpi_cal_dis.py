import itertools
import json

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from sklearn.preprocessing import MinMaxScaler

import configs
from utils import *

CONFIG_PATH = 'configs'


def deal_if_zero(data):
    if np.all(data == 0):
        data = np.ones_like(data) * 1e-9
    return data


def pearsondis(x, y):
    index_num = x.shape[0]
    return index_num - np.sum(
        np.sum(np.multiply(x, y), axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))


def sbd(x, y):
    norm_cc_list = []
    index_num, m = x.shape
    x_norm = np.linalg.norm(x, axis=1)
    for s in range(-m + 1, m):
        tmp_norm_cc_list = []
        if s >= 0:
            # y_s = np.hstack((np.zeros((index_num, s)), y[:, :m - s]))
            tmp_norm_cc_list.append(
                np.sum(np.multiply(x[:, s:], y[:, :m - s]), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)))
        else:
            # y_s = np.hstack((y[:, -s:], np.zeros((index_num, -s))))
            tmp_norm_cc_list.append(
                np.sum(np.multiply(x[:, :m + s], y[:, -s:]), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)))
        norm_cc = np.sum(tmp_norm_cc_list)
        norm_cc_list.append(norm_cc)
    ncc = np.max(norm_cc_list)
    sbd = index_num - ncc
    return sbd


def euc(x, y):
    return np.sum(np.linalg.norm(x - y, axis=1))


def norm_euc(x, y):
    return np.sum(np.sqrt(np.sum(np.multiply(x - y, x - y, ), axis=1) / (
            np.sum(np.multiply(x, x), axis=1) * np.sum(np.multiply(y, y), axis=1))))


def Manhattan(x, y):
    return np.sum(np.linalg.norm(x - y, ord=1))


def Square(x, y):
    return np.sum(np.power((x - y), 2))


@time_consuming("CAL DISTANCE")
def cal_dis_matrix(data, comm, affinity="SBD"):
    rank = comm.Get_rank()
    size = comm.Get_size()
    index_comb = itertools.combinations(np.arange(data.shape[0]), 2)
    index_comb = np.array([i for i in index_comb])
    if affinity == "PearsonDistance":  # Distance: 1 - Pearson Correlation Coefficient
        dis_func = pearsondis
        mean_data = np.mean(data, axis=2)
        mean_data = np.expand_dims(mean_data, axis=2)
        data = data - mean_data
    elif affinity == "SBD":  # Distance: 1 - NCC  Smaller SBD means higher shape similarity
        dis_func = sbd
    elif affinity == "Euclidean":
        dis_func = euc
    elif affinity == "NormEuclidean":
        dis_func = norm_euc
    elif affinity == "Manhattan":
        dis_func = Manhattan
    elif affinity == "Square":
        dis_func = Square
    data_sum = np.sum(data, axis=2)
    zero_index = np.where(data_sum == 0)
    data[zero_index] = 1e-9
    distance_matrix = None
    if rank == 0:
        distance_matrix = np.zeros((data.shape[0], data.shape[0]))
        for _ in range(size - 1):
            recv_data = comm.recv(source=ANY_SOURCE)
            for (i, j, dis) in recv_data:
                distance_matrix[i][j] = dis
                distance_matrix[j][i] = dis
    else:
        index_comb_len = len(index_comb) // size
        if rank == size - 1:
            rank_index_comb = index_comb[(rank - 1) * index_comb_len:]
        else:
            rank_index_comb = index_comb[(rank - 1) * index_comb_len:rank * index_comb_len]
        i_j_dis = []
        for i, j in rank_index_comb:
            dis = dis_func(data[i], data[j])
            i_j_dis.append((i, j, dis))
        comm.send(i_j_dis, dest=0)
    return distance_matrix


def data_stand(data):
    data = np.transpose(data, (0, 2, 1))
    scaler = MinMaxScaler()
    data = [scaler.fit_transform(i) for i in data]
    data = np.transpose(data, (0, 2, 1))
    return data


def main(params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    data = np.squeeze(np.load(params["data_path"]))
    for dis in params["dis_list"]:
        distance_matrix = cal_dis_matrix(data, comm, dis)
        comm.barrier()
        if rank == 0:
            np.save(params["distance_dir"].joinpath(f"{dis}.npy"), distance_matrix)


if __name__ == '__main__':
    config = configs.config(CONFIG_PATH)
    paras = {"data_path": config.get('feature_selection', 'data_save_path'),
             "dis_list": json.loads(config.get('cal_dis', 'dis_list')),
             "distance_dir": pathlib.Path(config.get('cal_dis', 'distance_dir'))}
    paras["distance_dir"].mkdir(parents=True, exist_ok=True)
    main(paras)
