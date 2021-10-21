import numpy as np

import configs
from utils import *

CONFIG_PATH = 'configs'


def diff_func(data, max_win):
    # equation (6)
    df = [0] * max_win
    if isinstance(data, list):
        data_length = len(data)
    elif isinstance(data, np.ndarray):
        data_length = data.shape[0]
    else:
        raise NotImplementedError("Unsupported data type")
    for win in range(1, max_win):
        for now_data in range(0, data_length - max_win):
            df[win] += np.power(data[now_data] - data[now_data + win], 2)
    return df


def get_cmndf(df):
    # equation (8)
    if isinstance(df, list):
        data_length = len(df)
    elif isinstance(df, np.ndarray):
        data_length = df.shape[0]
    else:
        raise NotImplementedError("Unsupported data type")
    cmndf = [1.0] * data_length
    for now_data in range(1, data_length):
        if np.sum(df[:now_data + 1]) != 0:
            cmndf[now_data] = df[now_data] * (now_data + 1) / np.sum(df[:now_data + 1]).astype(float)
    return cmndf


def get_pitch(cmndf, min_win, max_win, th):
    for win in range(min_win, max_win):
        if cmndf[win] < th:
            if win + 1 < max_win and cmndf[win + 1] < cmndf[win]:
                continue
            return win
    return 0


def get_yin(data, win_min, win_max, th):
    # Compute YIN
    if np.all(np.array(data) == 0):
        return 0
    df = diff_func(data, win_max)
    if np.all(np.array(df) == 0):
        return 0
    cmndf = get_cmndf(df)
    p = get_pitch(cmndf, win_min, win_max, th)
    return data.shape[0] // p if p != 0 else 0


@time_consuming("GET CYCLE")
def get_all_cycle(data, win_min, win_max, th):
    cycle = []
    for index_ins, data_ins in enumerate(data):
        for index_index, data_index in enumerate(data_ins):
            cycle.append([index_ins, index_index,
                         get_yin(data_index, win_min, win_max, th)])
    return np.array(cycle)


def main(params):
    data = np.squeeze(np.load(params["data_path"]))
    cycle = get_all_cycle(data, params["win_min"], params["win_max"], params["th"])
    np.save(params["cycle_path"], cycle)


if __name__ == "__main__":
    config = configs.config(CONFIG_PATH)
    paras = {"data_path": config.get('train_ae', 'init_ae_z_file'),
             "cycle_path": pathlib.Path(config.get('yin', 'cycle_path')),
             "win_min": config.getint('yin', 'win_min'),
             "win_max": config.getint('yin', 'win_max'),
             "th": config.getfloat('yin', 'th'),
             }
    paras["cycle_path"].parent.mkdir(parents=True, exist_ok=True)
    main(paras)
