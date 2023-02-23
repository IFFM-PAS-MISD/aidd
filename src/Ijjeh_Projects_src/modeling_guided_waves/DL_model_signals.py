import os
import numpy as np
import json
import csv
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from decouple import config
from sklearn.metrics import mean_squared_error

access_token = config('NEPTUNE_API_TOKEN')

########################################################################################################################
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['DL signal based modelling', 'MSE_Fourier_domain']
                       )

# hyperparameters
run["Signal_based"] = "AE"
params = {'samples': 486400,
          'n_size': 389120,
          'normalised': True,
          'batches': 128,
          'num_filters': 32,
          'kernel_size': 3,
          'shape': (512, 1),
          'epochs': 10000,
          'dropout': 0.2,
          'levels': 9,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96
          }

run["model/parameters"] = params

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

########################################################################################################################
env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'
os.chdir(env_path)
json = json.dumps(params)
f = open('hyper_par_GWM_mse.json', 'w')
f.write(json)
f.close()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    params['learning_rate'],
    decay_steps=10000,
    decay_rate=0.96,
    staircase=False)


def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    X_train_1 = np.load('LR_GT_del.npy')
    X_train_1 = np.reshape(X_train_1, (475, 5, 1))
    X_train_1 = np.transpose(X_train_1, [0, -1, 1])
    X_train_1 = np.repeat(X_train_1, 1024, axis=1)
    X_train_1 = np.reshape(X_train_1, (1024 * 475, 5))
    print(X_train_1.shape)

    X_train_2 = np.load('LR_ref_frames.npy')
    print(X_train_2.shape)
    Y_train = np.load('LR_labels.npy')
    print(Y_train.shape)

    x_train_1 = X_train_1[:params['n_size']]
    x2_ref_train = X_train_2[:params['n_size']]
    y_in_train = Y_train[:params['n_size']]

    # for count in range(x1_train.shape[0]):
    #     print(count)
    #
    #     x = x1_train[count][0]
    #     y = x1_train[count][1]
    #     a = x1_train[count][2]
    #     b = x1_train[count][3]
    #     theta = x1_train[count][-1]
    #
    #     theta = (theta * 2 * np.pi) / 360  # to rad
    #
    #     f0 = theta
    #
    #     t = np.linspace(0, 1, 512)
    #
    #     x_cords_Amp = (a / b) * np.sqrt((x - 0.25) ** 2 + (y - 0.25) ** 2)
    #
    #     x_cords_modulated = x_cords_Amp * (np.cos(2 * np.pi * f0 * t))
    #
    #     x2_ref_train[count] = x2_ref_train[count] * x_cords_modulated
    #
    return x_train_1, x2_ref_train, y_in_train


Input_B, Input_A, Y = load_dataset()


class SpectralConv1d(tf.keras.layers.Layer):
    def __init__(self, in_channels=1, out_channels=32, modes1=128):
        super(SpectralConv1d, self).__init__()
        self.kernel = None
        """ 1D Fourier layer. It does FFT, linear transform, and Inverse FFT. """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,
                                                             dtype=torch.cfloat))

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


def custom_loss(y_true, y_pred):
    y_true_k = y_true
    y_pred_k = y_pred

    fft2d_true = tf.signal.fft(tf.cast(y_true_k, dtype=tf.complex64))
    fft2d_pred = tf.signal.fft(tf.cast(y_pred_k, dtype=tf.complex64))
    N = tf.size(fft2d_true)
    N = tf.cast(N, dtype=tf.float32)

    fft2d_true = tf.signal.fftshift(fft2d_true, axes=(-1))
    fft2d_pred = tf.signal.fftshift(fft2d_pred, axes=(-1))

    fft2d_true = tf.cast(fft2d_true, dtype=tf.float32)
    fft2d_pred = tf.cast(fft2d_pred, dtype=tf.float32)

    fft2d_pred = tf.divide(fft2d_pred, N)
    fft2d_true = tf.divide(fft2d_true, N)

    MSE_Fourier_domain = tf.losses.MSE(abs(fft2d_true), abs(fft2d_pred))
    MSE_Spatial = tf.losses.MSE(y_true, y_pred)

    return MSE_Fourier_domain  # + MSE_Spatial


def conv_block(in_blk, num, k_size):
    layer11 = tf.keras.layers.Conv1D(num,
                                     k_size,
                                     strides=1,
                                     padding='same',
                                     activation='relu')(in_blk)
    # layer11 = tf.keras.layers.Dropout(dropout)(layer11)
    # layer11 = tf.keras.layers.Conv1D(num,
    #                                  k_size,
    #                                  padding='same',
    #                                  activation='relu')(layer11)
    return layer11


def build_model():
    inputA = tf.keras.layers.Input(params['shape'], name='LR_img_input')
    ####################################################################
    ####################################################################
    encoder = conv_block(inputA, params['num_filters'], params['kernel_size'])
    skip_tensor = []
    for i in range(params['levels']):
        encoder = conv_block(encoder, params['num_filters'], params['kernel_size'])  #
        encoder = tf.keras.layers.MaxPool1D(name='max-pooling_encoder_1d_%d' % i)(encoder)
        skip_tensor.append(encoder)
    print(encoder)
    ####################################################################
    ####################################################################
    InputB = tf.keras.layers.Input(shape=(5,), name='input_damage_info')
    encoder = tf.keras.layers.Flatten()(encoder)
    inds = tf.keras.layers.Flatten()(InputB)
    Bottle_neck = tf.keras.layers.concatenate([encoder, inds], axis=-1)

    Bottle_neck = tf.keras.layers.Dense(Bottle_neck.shape[-1], activation='relu')(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dense(Bottle_neck.shape[-1] * 3, activation='relu')(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dropout(params['dropout'])(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dense(Bottle_neck.shape[-1] * 6, activation='relu')(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dropout(params['dropout'])(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dense(Bottle_neck.shape[-1] * 3, activation='relu')(Bottle_neck)
    Bottle_neck = tf.keras.layers.Dropout(params['dropout'])(Bottle_neck)
    decoder = tf.keras.layers.Dense(1024, activation='sigmoid')(Bottle_neck)
    ####################################################################
    ####################################################################
    decoder = tf.keras.layers.Reshape((1, decoder.shape[-1]))(decoder)
    for j in (range(1, params['levels'] + 1)):
        decoder = tf.keras.layers.concatenate((decoder, skip_tensor[-j]), axis=-1)
        decoder = tf.keras.layers.UpSampling1D(2)(decoder)
        decoder = conv_block(decoder, params['num_filters'], params['kernel_size'])
    output = tf.keras.layers.Conv1D(1, 1, padding='same', activation='sigmoid')(decoder)

    #
    ########################################################################################################################
    # Output layer
    ########################################################################################################################

    # layer_A = tf.signal.fft(tf.cast(inputA, dtype=tf.complex64))
    # layer_A = tf.signal.fftshift(layer_A)
    # layer_A = abs(layer_A)
    # layer_A = tf.math.l2_normalize(layer_A.numpy())

    # layer_A = tf.keras.layers.Dense(512, activation='relu')(inputA)
    # for i in range(levels):
    #     layer_A = tf.keras.layers.Dense(512, activation='relu')(layer_A)
    #     layer_A = tf.keras.layers.Dropout(dropout)(layer_A)
    # # flat_layer = tf.keras.layers.Flatten()(layer_A)
    #
    # concat_layer = tf.keras.layers.concatenate([inputB, layer_A], axis=-1)
    # output = tf.keras.layers.Dense(516, activation='relu')(concat_layer)
    # output = tf.keras.layers.Dropout(dropout)(output)
    # output = tf.keras.layers.Dense(768, activation='relu')(output)
    # output = tf.keras.layers.Dropout(dropout)(output)
    # output = tf.keras.layers.Dense(512, activation='sigmoid')(output)

    # define a model with a list of two inputs
    # print(output)
    # model3 = tf.keras.Model(inputs=[inputs_2], outputs=output, name='model_3')
    # model3.summary()
    ####################################################################

    # vae_input1 = tf.keras.layers.Input(shape=(512, 1), name="VAE_input1")
    # vae_input2 = tf.keras.layers.Input(shape=(4,), name="VAE_input2")

    # vae_decoder_output = model3(model2([model1(inputA), inputB]))
    model = tf.keras.models.Model([inputA, InputB], output, name='AE_model')

    model.summary()
    model.compile(tf.keras.optimizers.Adam(lr_schedule),
                  loss='mse',
                  metrics=[tf.keras.metrics.CosineSimilarity()],
                  run_eagerly=False)
    return model


checkpoint_filepath = env_path + 'temp/checkpoint/Signal_based_modelling_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    params['samples'],
    params['learning_rate'],
    params['num_filters'],
    params['levels'],
    params['batches'],
    params['epochs'],
    params['dropout'],
    params['val_split'],
    params['hidden_layer'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=params['patience_epochs'],
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run[metric_name].log(metric_value)


AE_model = build_model()
AE_model.fit(x=[Input_A, Input_B],
             y=Y,
             batch_size=params['batches'],
             validation_split=params['val_split'],
             epochs=params['epochs'],
             callbacks=[callbacks, MonitoringCallback()])
