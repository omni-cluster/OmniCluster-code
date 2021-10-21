import itertools
from collections import Counter

import numpy as np
import copy
import configs
from utils import *

CONFIG_PATH = 'configs'


@time_consuming("feature selection: GET CYCLE DATA")
def get_cycle_data(path):
    """
    读取周期性数据文件
    Args:
        path: 周期性数据文件，文件内数据形状为 (items_num * index_num, 3)
              数据id，数据index，数据周期性
    Returns:
        index_cycle: index周期性的对应关系，形状为 (items_num * index_num, 2)
                     数据index，数据周期性
    """
    cycle = np.load(path)
    return cycle[:, 1:]

@time_consuming("feature selection: GET INDEX ABOVE TH")
def get_index_equal_cycle_above_thred(index_cycle, th):
    """
    筛选周期性等于带保留周期性的指标，并计算每个指标具有周期性的次数
    Args:
        index_cycle: index周期性的对应关系，形状为 (items_num * index_num, 2)
                     数据index，数据周期性
        cycle_list: 待保留的周期性

    Returns:
        index_list: 该指标具有周期性数据量超过指定阈值

    """
    index_freq = {}
    index_list = []
    for i, j in index_cycle:
        if j > 0:
            # 只要周期数不是0，就代表具有周期性
            if i in index_freq:
                index_freq[i] += 1
            else:
                index_freq[i] = 1
    for i, j in index_freq.items():
        if j > th:
            index_list.append(i)
    # 对index进行排序，便于后续对比各个指标之间的相似性，但也去除了筛选一对相似指标时的随机性
    index_list.sort()
    return index_list


@time_consuming("feature selection: GET ORIGIN DATA")
def get_origin_data(path, index_list):
    """
    读取原始数据文件
    Args:
        path: 原始数据文件路径
        index_list: 指定的index

    Returns:
        data: 指定index的原始数据
    """
    data = np.load(path)[:, index_list].squeeze()
    return data


@time_consuming("feature selection: CAL CORRELATION")
def cal_correlation(data, index_comb):
    """
    计算指标之间的相关度
    Args:
        data: 数据
        index_comb: 指标组合

    Returns:
        data_correlation: 数据内部指标间相关程度，形状为(items_num, 指标组合数量)
    """
    mean_data = np.mean(data, axis=2)
    mean_data = np.expand_dims(mean_data, axis=2)
    data = data - mean_data
    index_correlation = []
    for i_data in data:
        tem_correlation = []
        for index in index_comb:
            x = i_data[index[0]]
            y = i_data[index[1]]
            if np.all(x == 0):
                x = np.ones_like(x) * 1e-9
            if np.all(y == 0):
                y = np.ones_like(y) * 1e-9
            cor = np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))
            tem_correlation.append(cor)
        index_correlation.append(np.array(tem_correlation))
    return np.array(index_correlation)

def cal_correlation(data, index_comb):
    """
    计算指标之间的相关度
    Args:
        data: 数据
        index_comb: 指标组合

    Returns:
        data_correlation: 数据内部指标间相关程度，形状为(items_num, 指标组合数量)
    """
    mean_data = np.mean(data, axis=2)
    mean_data = np.expand_dims(mean_data, axis=2)
    data = data - mean_data
    index_correlation = []
    for i_data in data:
        tem_correlation = []
        for index in index_comb:
            x = i_data[index[0]]
            y = i_data[index[1]]
            if np.all(x == 0):
                x = np.ones_like(x) * 1e-9
            if np.all(y == 0):
                y = np.ones_like(y) * 1e-9
            cor = np.sum(
                np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)))
            tem_correlation.append(cor)
        index_correlation.append(np.array(tem_correlation))
    return np.array(index_correlation)

def meet_rule1(R, F, index_list, F_index_list):
    # 判断R中是否存在某一个i与其他行都不相关，R对应位置为0
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if np.sum(R[i,tmp_F_index_list]) == 0:
            return index_list[i], True
    return -1, False

def meet_rule2(R, F, index_list, F_index_list):
    # 判断R中是否存在某一个i与其他指标都相关，但此时别的指标之间存在不相关的情况
    # 1.先判断R中是否存在0，存在的话再进行下一步
    exist_0_flag = False
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if not np.all(R[i, tmp_F_index_list]):
            exist_0_flag = True
            break
    if not exist_0_flag:
        return -1, False
    # 2.判断是否存在i与其他所有指标都相关
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if np.all(R[i, tmp_F_index_list]):
            return index_list[i], True
    return -1, False

def meet_rule3(R, F, index_list, F_index_list):
    # 判断是否R中所有的指标都互相相关
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if not np.all(R[i, tmp_F_index_list]):
            return False
    return True

@time_consuming("feature selection: GET INDEX ABOVE CORRELATION THRED")
def get_index_above_corre_thred(index_comb, data_correlation, sim_th, sim_num_th, index_list, data_num):
    """
    对指标间相似程度大于`sim_th`且在所有数据中，两个指标出现相似性的数量大于`sim_num_th`的指标去除冗余
    Args:
        index_comb: 指标组合
        data_correlation: 数据内部指标间相关程度，形状为(items_num, 指标组合数量)
        sim_th: 相似度阈值
        sim_num_th: 相似数量阈值
        index_list: 周期性筛选出的指标列表
        data_num: 数据总量

    Returns:
        index_to_save: 不具有冗余的指标
    """
    cor = (data_correlation > sim_th).astype(int)
    cor = np.sum(cor, axis=0)
    # 计算相似度矩阵
    R = np.zeros((len(index_list), len(index_list)))
    for cor_item, (j, i) in zip(cor, index_comb):
        # 将没有超过阈值的冗余矩阵的值设置为0
        if cor_item > sim_num_th:
            # print(f"{index_list[j]}和{index_list[i]}指标相似度较高，去掉{index_list[i]}指标, 相似度: {cor_item}")
            # index_exp.append(i)
            R[i][j] = cor_item
            R[j][i] = cor_item
    # 将R转化为百分数
    R = R / data_num
    # 按照规则去除冗余的特征
    F = set(index_list) # 具有周期性的所有指标
    select_F = set() # 被选择的指标
    delete_F = set() # 被删除的指标
    # index_list 指标和索引的对应关系
    while len(F) > 0:
        print(f"F:{F}\nSF:{select_F}\nDF:{delete_F}")
        print('-'*20)
        F_index_list = [index_list.index(f) for f in F]
        select_F_index_list = [index_list.index(s_f) for s_f in select_F]
        # 判断规则1
        i, if_continue = meet_rule1(R, F, index_list, F_index_list)
        if if_continue:
            print('meet_rule1')
            # rule1的处理
            print(f"选择指标:{i}")
            F.discard(i)
            select_F.add(i)
            continue
        # 判断规则2
        i, if_continue = meet_rule2(R, F, index_list, F_index_list)
        if if_continue:
            print('meet_rule2')
            print(f"删除指标:{i}")
            # rule2的处理
            F.discard(i)
            delete_F.add(i)
            continue
        if meet_rule3(R, F, index_list, F_index_list):
            print('meet_rule3')
            # rule3的处理
            # 3.1 选择i和当前SF中相关度最小的
            # 如果当前SF为空，i为F的第一个
            if len(select_F) == 0:
                i = list(F)[0]
                print(f"SF集合为空，选择指标:{i}")
            else:
                min_correlation_i_SF_sum = 100
                for f_index in F_index_list:
                    tmp_sum = np.sum(R[f_index, select_F_index_list])
                    if  tmp_sum < min_correlation_i_SF_sum:
                        min_correlation_i_SF_sum = tmp_sum
                        i = index_list[f_index]
                print(f"选择指标:{i}")
            # 3.2 i加入SF，F中删除i
            select_F.add(i)
            F.discard(i)
            # 3.3 将F中的剩余元素加入DF，清除F，结束整个迭代
            print(f"将F中剩余元素都删除，F:{F}")
            delete_F = delete_F | F
            F.clear()
            break
        else:
            print('meet_rule4')
            # 4.1 选出i与F中指标相关性最小的
            min_correlation_i_F_sum = 100
            i_index = None
            for f_index in F_index_list:
                tmp_sum = np.sum(R[f_index, F_index_list])
                if tmp_sum < min_correlation_i_F_sum:
                    min_correlation_i_F_sum = tmp_sum
                    i = index_list[f_index]
                    i_index = f_index
            print(f"i与F中指标相关性最小, 选择指标i:{i}")
            # 4.2 S_i为F中与i相关的特征子集，再选出S_i中的j与SF最相关的
            S_i = set()
            for f_index in F_index_list:
                # F中当前遍历的元素不是i，且i与之相关
                if index_list[f_index] != i and R[i_index, f_index] > 0:
                    # 将该指标加入S_i
                    S_i.add(index_list[f_index])
            print(f"F集合中和指标i相关子集S_i:{S_i}")
            max_correlation_j_SF_sum = 0
            S_i_index_list = [index_list.index(s) for s in S_i]
            j = None
            for s_index in S_i_index_list:
                tmp_sum = np.sum(R[s_index, select_F_index_list])
                if tmp_sum > max_correlation_j_SF_sum:
                    print(f"4.2 SF:{select_F} max_correlation_j_SF_sum:{max_correlation_j_SF_sum} tmp_sum:{tmp_sum}")
                    max_correlation_j_SF_sum = tmp_sum
                    j = index_list[s_index]
            if j is None:
                # SF中的指标和S_i中的指标相关性为0，没有筛选出j,这时选择S_i中最小的j（自己定的）
                j = min(S_i)
            print(f"S_i中的j与SF相关性最大,删除j:{j}")
            # 4.3 i加入SF，j加入DF，F中删除i
            select_F.add(i)
            delete_F.add(j)
            F.discard(i)
            F.discard(j)
            # 继续循环
    print(f"F:{F}\nSF:{select_F}\nDF:{delete_F}")
    # 输出原始的冗余矩阵
    print('row', end='\t')
    for index in index_list:
        print(index, end='\t')
    print()
    for row_index, row in enumerate(R):
        print(index_list[row_index], end='\t')
        for col_index, cor in enumerate(row):
            if cor > sim_num_th / 5711:
                print(f"\033[0;31m{round(cor, 3)}\033[0m", end='\t')
            else:
                print(round(cor, 3), end='\t')
        print()
    return list(select_F)




def main(params):
    index_cycle = get_cycle_data(params["cycle_path"])
    index_list = get_index_equal_cycle_above_thred(index_cycle, params["index_th"])
    data = get_origin_data(params["data_path"], index_list)
    index_comb = itertools.combinations(np.arange(data.shape[1]), 2)
    index_comb = np.array([i for i in index_comb])
    data_correlation = cal_correlation(data, index_comb)
    index_to_save = get_index_above_corre_thred(index_comb, data_correlation, params["theta_s"],
                                                params["theta_p"], index_list, data.shape[0])
    data_to_save = data.squeeze()[:, index_to_save, :]
    np.save(params["data_save_path"], data_to_save)


if __name__ == "__main__":
    config = configs.config(CONFIG_PATH)
    paras = {"data_path": config.get('train_ae', 'init_ae_z_file'),
             "cycle_path": pathlib.Path(config.get('yin', 'cycle_path')),
             "theta_y": config.getint('feature_selection', 'index_th'),
             "theta_s": config.getfloat('feature_selection', 'sim_th'),
             "theta_p": config.getint('feature_selection', 'sim_num_th'),
             "data_save_path": pathlib.Path(config.get('feature_selection', 'data_save_path'))
             }
    paras["data_save_path"].parent.mkdir(parents=True, exist_ok=True)
    main(paras)
