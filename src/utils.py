import csv
import os
import pathlib
from time import time

import sklearn.cluster as sc
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import models, layers, Model

from cluster import *


def time_consuming(period):
    def real_time_consuming(func):
        def cal(*args):
            start = time()
            res = func(*args)
            print(period, "time costing:", time() - start, "\n")
            return res

        return cal

    return real_time_consuming


@time_consuming("LOAD TRAIN DATA")
def get_train_data(file):
    x_train = np.load(pathlib.Path(file))
    return x_train


@time_consuming("GET Z")
def get_save_z(data, model_file, mode, save_file=None):
    model = models.load_model(
        model_file,
        custom_objects={"Conv_1D": Conv_1D, "Conv_1D_Transpose": Conv_1D_Transpose},
    )
    if mode == "softmax":
        encoder_model = Model(
            inputs=model.input[0], outputs=model.get_layer("z").output
        )
    else:
        encoder_model = Model(inputs=model.input, outputs=model.get_layer("z").output)
    z = encoder_model.predict(data)
    if save_file:
        np.save(save_file, z)
    return z


@time_consuming("GET RECONSTRUCT")
def get_save_reconstruct(data, model_file, mode, save_file=None):
    model = models.load_model(
        model_file,
        custom_objects={"Conv_1D": Conv_1D, "Conv_1D_Transpose": Conv_1D_Transpose},
    )
    if mode == "softmax":
        model = Model(
            inputs=model.input[0], outputs=model.get_layer("recons_out").output
        )
    else:
        model = Model(inputs=model.input, outputs=model.get_layer("recons_out").output)
    reconstruct = model.predict(data)
    if save_file:
        np.save(save_file, reconstruct)
    return reconstruct


def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a += 1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b += 1
            else:
                pass
    ri = (a + b) / (n * (n - 1) / 2)
    return ri


def get_metrics(y_true, y_pred):
    ri = rand_index(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)  # -1~1 1
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)  # -1~1 1
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)  # -1~1 1
    h = metrics.homogeneity_score(y_true, y_pred)
    c = metrics.completeness_score(y_true, y_pred)
    v = metrics.v_measure_score(y_true, y_pred)  # 0-1 1
    fmi = metrics.fowlkes_mallows_score(y_true, y_pred)  # 0-1 1
    return {
        "RI": ri,
        "ARI": ari,
        "NMI": nmi,
        "AMI": ami,
        "H": h,
        "C": c,
        "V": v,
        "FMI": fmi,
    }


def write_ana(writer, first_row, data):
    writer.writerow(first_row)
    for i_, i_data in enumerate(data):
        to_write_data = [i_, [_ for _ in i_data]]
        writer.writerow(to_write_data)
    writer.writerow([])
    return


@time_consuming("LOAD PREDICT DATA")
def get_predict_data(file_dir):
    x_predict = np.load(os.path.join(file_dir, "predict_data.npy"))
    if os.path.exists(os.path.join(file_dir, "predict_label.npy")):
        y_predict = np.load(os.path.join(file_dir, "predict_label.npy"))
    else:
        y_predict = None
    return (x_predict, y_predict)


@time_consuming("CLUSTERING")
def fit_predict_cluster(train_data, test_data, n_cluster=None, mode=1, linkage="average", threshd=10):
    train_num = train_data.shape[0]
    data = np.vstack((train_data, test_data))
    data = data.reshape((data.shape[0], -1))
    if mode == 1:
        clu_model = sc.AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
        clu_label = clu_model.fit_predict(data)
    elif mode == 2:
        _, clu_label, _, _ = kmeans_cluster(data, n_cluster)
    elif mode == 3:
        _, clu_label = gac_cluster(data, n_cluster)
    # elif mode == 4:
    #     clu_model = sc.AgglomerativeClustering(n_clusters=None, linkage=linkage, distance_threshold=threshd)
    #     clu_label = clu_model.fit_predict(data)
    # elif mode == 5:
    #     clu_model = sc.AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage, affinity="precomputed")
    #     clu_label = clu_model.fit_predict(data)
    clu_label_train = clu_label[:train_num]
    clu_label_test = clu_label[train_num:]
    return clu_label_train, clu_label_test


@time_consuming("GET PREDICT")
def get_predict(data_to_predict, model_path, predict_type):
    model = models.load_model(
        model_path,
        custom_objects={"Conv_1D": Conv_1D, "Conv_1D_Transpose": Conv_1D_Transpose},
    )
    if predict_type == "init_softmax":
        model = Model(
            inputs=model.input[0], outputs=model.get_layer("prediction").output
        )
    elif predict_type == "final_model":
        model = Model(inputs=model.input, outputs=model.get_layer("prediction").output)
    predict = model.predict(data_to_predict)
    predict = [np.argmax(x) for x in predict]
    return predict


def cal_save_metrics(y_true, y_pred, file_path, period="", clear=False):
    if clear and os.path.exists(file_path):
        os.remove(file_path)
    metrics_ = get_metrics(y_true, y_pred)
    with open(file_path, "a") as w:
        w.write(period)
        for metrics_name, metrics_value in metrics_.items():
            w.write(metrics_name + ":" + str(metrics_value) + "\n")
        w.write("\n")


def analyze(n_label, n_cluster, label_file, pred_file, save_file):
    label = np.load(label_file)
    pred = np.load(pred_file)
    sta_data = [([0] * n_cluster) for _ in range(n_label)]  # shape:(label,predict)
    c = [([0] * n_cluster) for _ in range(n_label)]
    r = [([0] * n_cluster) for _ in range(n_label)]
    for i_label in range(1, n_label + 1):
        _pred = pred[np.where(label == i_label)]
        for n_cluster in range(n_cluster):
            sta_data[i_label - 1][n_cluster] = np.where(_pred == n_cluster)[0].shape[0]
    c_all = np.sum(sta_data, axis=1)
    r_all = np.sum(sta_data, axis=0)
    for i_, c_data in enumerate(sta_data):
        for j_ in range(c_data.shape[0]):
            if int(c_all[i_]) == 0:
                c[i_][j_] = 0
            else:
                c[i_][j_] = sta_data[i_][j_] / int(c_all[i_])
            if int(r_all[j_]) == 0:
                r[i_][j_] = 0
            else:
                r[i_][j_] = sta_data[i_][j_] / int(r_all[j_])

    result_file = open(save_file, "w")
    writer = csv.writer(result_file)
    writer.writerow(["n_cluster:" + str(n_cluster) + "; n_label"] + str(n_label))
    first_row = []
    first_row.append("label/preidct:")
    for i_ in range(n_cluster):
        first_row.append(i_)
    write_ana(writer, first_row, sta_data)
    write_ana(writer, first_row, c)
    write_ana(writer, first_row, r)
    result_file.close()


# def deal_p_prob(data):
#     """
#     deal the total probability of each label
#
#     Arguments:
#         data: numpy array, the probabilities on each label of each input
#
#     Returns:
#         data: numpy array, the probabilities on each label of the input
#     """
#     cluster_frequency = np.sum(data, axis=0)
#     data = data ** 2 / cluster_frequency
#     data = np.transpose(data.T / np.sum(data, axis=1))
#     return data
#
#
# def iterate_batchs(data, batch_size, shuffle=True):
#     """
#     divide the input data into batches
#
#     Arguments:
#         data: numpy array, input data
#         batch_size: int, the size of each batch needed to be divided into
#         shuffle: bool, to judge if shuffle the order of the input data in each batch
#
#     Returns:
#         X[index]: numpy array, data divided into batch
#         index: list, the index of the data in batch
#     """
#     if shuffle:
#         indices = np.arange(len(data))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(data), batch_size):
#         if start_idx + batch_size < len(data):
#             end_idx = start_idx + batch_size
#         else:
#             end_idx = len(data)
#         if shuffle:
#             index = indices[start_idx:end_idx]
#         else:
#             index = slice(start_idx, end_idx)
#         yield data[index], index


class Conv_1D(layers.Layer):
    def __init__(self, filters, stride, kernel_size, activation, **kwargs):
        super(Conv_1D, self).__init__()
        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.layers = []

    def build(self, input_shape):
        self.index_dims = input_shape[1]
        for i in range(self.index_dims):
            self.layers.append(
                layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    activation=self.activation,
                    input_shape=input_shape[2:],
                )
            )
        super(Conv_1D, self).build(input_shape)

    def call(self, inputs):
        out = []
        for i in range(self.index_dims):
            out.append(tf.expand_dims(self.layers[i](inputs[:, i, :, :]), 1))
        output = tf.concat(out, 1)
        return output

    def get_config(self):
        config = {
            "filters": self.filters,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
        }
        base_config = super(Conv_1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv_1D_Transpose(layers.Layer):
    def __init__(
            self,
            filters,
            stride,
            kernel_size,
            activation,
            name,
            output_padding=(0),
            **kwargs
    ):
        super(Conv_1D_Transpose, self).__init__(name=name)
        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.activation = activation
        self.layers = []

    def build(self, input_shape):
        self.index_dims = input_shape[1]
        for i in range(self.index_dims):
            self.layers.append(
                layers.Conv1DTranspose(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    activation=self.activation,
                    input_shape=input_shape[2:],
                    output_padding=self.output_padding,
                )
            )
        super(Conv_1D_Transpose, self).build(input_shape)

    def call(self, inputs):
        out = []
        for i in range(self.index_dims):
            out.append(tf.expand_dims(self.layers[i](inputs[:, i, :, :]), 1))
        output = tf.concat(out, 1)
        return output

    def get_config(self):
        config = {
            "filters": self.filters,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "output_padding": self.output_padding,
            "activation": self.activation,
        }
        base_config = super(Conv_1D_Transpose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class En_dense(layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(En_dense, self).__init__(**kwargs)
#         self.units = units
#         self.dense = layers.Dense(units=self.units, activation="relu")
#
#     def call(self, inputs):
#         data = tf.transpose(inputs, [0, 2, 1])
#         out = self.dense(data)
#         return out
#
#     def get_config(self):
#         base_config = super(En_dense, self).get_config()
#         config = {"units": self.units}
#         return {**base_config, **config}
#
#
# class De_dense(layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(De_dense, self).__init__(**kwargs)
#         self.units = units
#         self.dense = layers.Dense(units=self.units, activation="relu")
#
#     def call(self, inputs):
#         data = self.dense(inputs)
#         out = tf.transpose(data, [0, 2, 1])
#         return out
#
#     def get_config(self):
#         base_config = super(De_dense, self).get_config()
#         config = {"units": self.units}
#         return {**base_config, **config}
