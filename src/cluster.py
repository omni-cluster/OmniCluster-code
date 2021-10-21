import configs
from utils import *
import numpy as np
import json

CONFIG_PATH = 'configs'


def main(params):
    label = np.load(params["label_file"])
    for dis in params["dis_list"]:
        distance_matrix = np.load(params["distance_dir"].joinpath(f"{dis}.npy"))
        start = time()
        pred = sc.AgglomerativeClustering(n_clusters=None, compute_full_tree=True, affinity="precomputed",
                                          linkage=params["linkage"],
                                          distance_threshold=params["distance_threshold"]).fit_predict(
            distance_matrix)
        np.save(params["result_dir"].joinpath(f"{dis}_pred.npy"), pred)
        print(f"cluster with {dis} consume {time() - start}s")
        metrics = get_metrics(label, pred)
        with open(params["result_dir"].joinpath(f"{dis}_metrics.txt"), "w") as w:
            for key, value in metrics.items():
                w.write(f"{key} : {value}\n")


if __name__ == '__main__':
    config = configs.config(CONFIG_PATH)
    paras = {"dis_list": json.loads(config.get('cluster', 'dis_list')),
             "distance_dir": pathlib.Path(config.get('cal_dis', 'distance_dir')),
             "linkage": config.get('cluster', 'linkage'),
             "distance_threshold": config.getfloat('cluster', 'distance_threshold'),
             "label_file": config.get('cluster', 'label_file'),
             "result_dir": pathlib.Path(config.get('cluster', 'result_dir'))
             }
    paras["result_dir"].mkdir(parents=True, exist_ok=True)
    main(paras)
