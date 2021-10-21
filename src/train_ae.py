import json

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dropout

import configs
from utils import *

CONFIG_PATH = 'configs'


@time_consuming("INIT AE MODEL")
def init_model(data_shape, layer_num, kernel_size_list, stride_list, output_padding_list, dropout_list, filters_list,
               activation, optimizer, loss):
    K.clear_session()
    model = tf.keras.Sequential()
    model.add(Input(shape=data_shape))
    for i in range(layer_num):
        model.add(Conv_1D(filters=filters_list[i], kernel_size=kernel_size_list[i], stride=stride_list[i],
                          activation=activation, name=f"encode_{i}"))
        if i == layer_num - 1:
            model.add(Dropout(dropout_list[i], name="z"))
        else:
            model.add(Dropout(dropout_list[i]))
    for i in range(layer_num):
        if i == layer_num - 1:
            model.add(
                Conv_1D_Transpose(filters=data_shape[2],
                                  kernel_size=kernel_size_list[layer_num - 1 - i],
                                  stride=stride_list[layer_num - 1 - i], output_padding=output_padding_list[i],
                                  activation=activation, name="recons_out"))
        else:
            model.add(
                Conv_1D_Transpose(filters=filters_list[layer_num - 2 - i],
                                  kernel_size=kernel_size_list[layer_num - 1 - i],
                                  stride=stride_list[layer_num - 1 - i], output_padding=output_padding_list[i],
                                  activation=activation, name=f"decode_{layer_num-1-i}"))

    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model


@time_consuming("TRAIN AE MODEL")
def train_ae_model(model, data, batch_size, epochs, validation_split, save_path):
    model.fit(data, data, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    model.save(save_path)


def main(params):
    x_train = np.expand_dims(get_train_data(params["data_path"]), 3)
    data_shape = x_train.shape[1:]
    model = init_model(data_shape, params["layers_num"], params["kernel_size_list"], params["stride_list"],
                       params["output_padding_list"], params["dropout_list"], params["filters_list"],
                       params["activation"],
                       params["optimizer"], params["loss"])
    train_ae_model(model, x_train, params["batch_size"], params["epochs"], params["validation_split"],
                   params["init_ae_model_file"])
    x_train_z = get_save_z(
        x_train,
        params["init_ae_model_file"],
        "ae",
        params["init_ae_z_file"]
    )
    x_train_recons = get_save_reconstruct(
        x_train,
        params["init_ae_model_file"],
        "ae",
        params["init_ae_recons_file"]
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = configs.config(CONFIG_PATH)
    paras = {"data_path": config.get('train_ae', 'data_path'),
             "record_path": pathlib.Path(config.get('train_ae', 'record_path')),
             "init_ae_model_file": pathlib.Path(config.get('train_ae', 'init_ae_model_file')),
             "init_ae_z_file": pathlib.Path(config.get('train_ae', 'init_ae_z_file')),
             "init_ae_recons_file": pathlib.Path(config.get('train_ae', 'init_ae_recons_file')),
             "layers_num": config.getint('train_ae', 'layers_num'),
             "kernel_size_list": json.loads(config.get('train_ae', 'kernel_size_list')),
             "stride_list": json.loads(config.get('train_ae', 'stride_list')),
             "output_padding_list": json.loads(config.get('train_ae', 'output_padding_list')),
             "filters_list": json.loads(config.get('train_ae', 'filters_list')),
             "dropout_list": json.loads(config.get('train_ae', 'dropout_list')),
             "activation": config.get('train_ae', 'activation'),
             "if_shuffle": config.getboolean('train_ae', 'if_shuffle'),
             "batch_size": config.getint('train_ae', 'batch_size'),
             "epochs": config.getint('train_ae', 'epochs'),
             "validation_split": config.getfloat('train_ae', 'validation_split'),
             "optimizer": config.get('train_ae', 'optimizer'),
             "loss": config.get('train_ae', 'loss'), }
    paras["record_path"].parent.mkdir(parents=True, exist_ok=True)
    paras["init_ae_model_file"].parent.mkdir(parents=True, exist_ok=True)
    paras["init_ae_z_file"].parent.mkdir(parents=True, exist_ok=True)
    paras["init_ae_recons_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(paras["record_path"], "w") as f:
        f.writelines("------------------ start ------------------\n")
        for eachArg, value in paras.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
        f.writelines("------------------- end -------------------\n")

    main(paras)
