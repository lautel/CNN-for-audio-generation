from __future__ import division
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Convolution1D, Merge, Activation, AtrousConvolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
from keras.regularizers import l2, activity_l2
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.engine import Model
from scipy.signal import chirp
from scipy.io import wavfile
import config.nn_config as nn_config
from IPython import embed

config = nn_config.get_parameters()

# Generate input waves
def generate_input(nwaves, f_set, ph_set, t, mode='simple'):
    Amax = 1
    wave_ = []

    for s in range(nwaves):
        if mode == 'simple':
            sine = Amax * np.sin(2 * np.pi * f_set[s] * t)

        elif mode == 'mixture1':
            # Mixture of 2 sinusoids at different frequencies
            sine1 = Amax * np.sin(2 * np.pi * f_set[0] * t)
            sine2 = Amax * np.sin(2 * np.pi * f_set[1] * t)
            sine = sine1 + sine2

        elif mode == 'mixture2':
            # Mixture of 3 sinusoids at different frequencies
            sine1 = Amax * np.sin(2 * np.pi * f_set[0] * t)
            sine2 = Amax * np.sin(2 * np.pi * f_set[1] * t)
            sine3 = Amax * np.sin(2 * np.pi * f_set[2] * t)
            sine = sine1 + sine2 + sine3

        elif mode == 'mixture3':
            # Mixture of sinusoids at different frequencies with random phases
            sine1 = Amax * np.sin(2 * np.pi * f_set[0] * t + ph_set[0])
            sine2 = Amax * np.sin(2 * np.pi * f_set[1] * t + ph_set[1])
            sine3 = Amax * np.sin(2 * np.pi * f_set[2] * t + ph_set[2])
            sine = sine1 + sine2 + sine3

        sine = sine / np.max(np.abs(sine))  # Normalize amplitude
        wave_.append(sine)

    return wave_


# Definition of quantizer
def quantifier(waves, Amax, nbits):
    xx_ = []
    Amin = -Amax
    ndecimal = nbits - 1
    q = np.round(((Amax - Amin) / (2 ** nbits)), decimals=ndecimal)
    # get all possible quantization levels
    val = np.arange(q / 2, Amax, q)
    values = np.sort(np.append(val, -val))

    for id in range(len(waves)):
        x = waves[id]
        qsignal = []
        for i in range(len(x)):
            aux = q * (np.floor(x[i] / q) + 0.5)
            # find the correspondent quantization level:
            idx = np.searchsorted(values, aux)
            if idx == len(values): idx -= 1
            qsignal.append(values[idx])
        xx_.append(qsignal)  # data input will be the signal generated without the last sample

    return [xx_, values]


## Redefine input data: one-hot encoding and reshape to the correct dimensions
def one_hot_encoding(X, values):
    one_hot_encode=[]

    for idx in range(len(X)):  # each tone generated
        tone = X[idx]
        # reset the matrix anytime we encode a new tone
        one_hot_mat = np.zeros((len(tone), len(values)), dtype=int)
        for j in range(len(tone)):  # each sample from the tone
            for i in range((len(values))):  # select the value among all possible values
                if tone[j] == values[i]:
                    one_hot_mat[j, i] = 1
                    break
        one_hot_encode.append(one_hot_mat)

    data_ = np.reshape(one_hot_encode, (len(one_hot_encode), one_hot_mat.shape[0], one_hot_mat.shape[1]))
    return data_


class printbatch(callbacks.Callback):
    # def on_batch_end(self, batch, logs={}):
    #     if batch%100 == 0:
    #         print "\nBatch " + str(batch) + " ends"

    def on_epoch_begin(self, epoch, logs={}):
        print(logs)

    def on_epoch_end(self, epoch, logs={}):
        print(logs)


def my_generator(datain, window, t):
    while 1:
        for ii in range(len(datain[0])-window-1):
            # Generation of input sub-dataset
            # if ii%50 == 0:
            #     randn = np.random.randint(low=-1, high=1)
            # else:
            #     randn = 0
            # ini = np.abs(ii+randn)
            data_train_ = datain[:,ii:ii+window, :]
            target_train_ = datain[:,ii+window+1, :]

            yield (data_train_, target_train_)


def my_generator_valid(datain, window, t):
    while 1:
        for ii in range(len(datain[0]) - window - 1):
            if ii%10 == 0:
                randn = np.random.randint(low=-1, high=1)
            else:
                randn = 0
            ini = np.abs(ii+randn)
            data_train_ = datain[:,ini:ini + window, :]
            target_train_ = datain[:,ini + window + 1, :]
            #data_train_ = np.reshape(data_train, (1, data_train.shape[0], data_train.shape[1]))
            #target_train_ = np.reshape(target_train, (1, target_train.shape[0]))

            yield (data_train_, target_train_)


def my_generator_test(datain, window, t):
    while 1:
        for ii in range(len(datain) - window - 1):
            # if ii % 10 == 0:
            #     randn = np.random.randint(low=-1, high=1)
            # else:
            #     randn = 0
            # ini = np.abs(ii + randn)
            data_train = datain[ii:ii + window, :]
            target_train = datain[ii + window + 1, :]

            data_train_ = np.reshape(data_train, (1, data_train.shape[0], data_train.shape[1]))
            target_train_ = np.reshape(target_train, (1, target_train.shape[0]))

            yield (data_train_, target_train_)


# Architecture #1 for the CNN
def baseline_model(data_, target_train_, nb_filters_, framerate_, loading='False', path=""):
    fnn_init = 'he_uniform'
    m = Sequential()
    m.add(
        Convolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            subsample_length=2,
            init = fnn_init,
            input_shape=(data_.shape[1], data_.shape[2]),
            activation='relu',
            # W_regularizer = l2(l=0.00001)
            # activity_regularizer=activity_l2(0.01)
        )
    )
    for k in range(int(np.log2(framerate_ - 1))):
        if m.output_shape[1] == 1:
            break
        else:
            m.add(
                Convolution1D(
                    nb_filter=nb_filters_,
                    filter_length=2,
                    subsample_length=2,
                    init = fnn_init,
                    activation='relu'
                )
            )

    m.add(Flatten())
    m.add(Dense(target_train_.shape[1], activation='softmax'))
    # Compile model
    # sgd = SGD(lr=0.01, momentum=0.8, decay=1e-4, nesterov=False)  # Values from https://arxiv.org/pdf/1512.07370.pdf
    ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, decay=0.0)
    if loading == 'True':
        m.load_weights(path)
    m.compile(loss='categorical_crossentropy', optimizer=ADAM, metrics=['accuracy'])
    return m


# Architecture #2 for the CNN
def baseline_model2(data_, target_, nb_filters_, framerate_, loading='False', path=""):
    fnn_init = 'he_uniform'
    m = Sequential()
    m.add(
        Convolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            init = fnn_init,
            input_shape=(data_.shape[1], data_.shape[2]),
            activation='relu',
            # W_regularizer = l2(l=0.00001)
            # activity_regularizer=activity_l2(0.01)
        )
    )
    for k in range(int(np.log2(np.floor(data_.shape[1])))):
        if m.output_shape[1] == 1:
            break
        else:
            m.add(
                Convolution1D(
                    nb_filter=nb_filters_,
                    filter_length=2,
                    subsample_length=2,
                    init = fnn_init,
                    activation='relu'
                )
            )

    m.add(Flatten())
    m.add(Dense(target_.shape[1], activation='softmax'))
    # Compile model
    # sgd = SGD(lr=0.01, momentum=0.8, decay=1e-4, nesterov=False)  # Values from https://arxiv.org/pdf/1512.07370.pdf
    ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, decay=0.0)
    if loading == 'True':
        m.load_weights(path)
        print "Weights loaded!"
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


# Architecture #3 for the CNN
def complex_model(data_, target_, nb_filters_, framerate_, loading='False', path=""):
    fnn_init = 'he_uniform'
    def residual_block(input_):

        tanh_ = Convolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            subsample_length=2,
            init=fnn_init,
            activation='tanh'
        )(input_)

        sigmoid_ = Convolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            subsample_length=2,
            init=fnn_init,
            activation='sigmoid'
        )(input_)

        out = Merge(mode='mul')([tanh_, sigmoid_])

        return out

    input = Input(shape=(data_.shape[1], data_.shape[2]), name='input_part')
    output = Convolution1D(
        nb_filter=nb_filters_,
        filter_length=2,
        init=fnn_init,
        activation='relu',
        border_mode='valid'
    )(input)

    for _ in range( int(np.log2(np.floor(data_.shape[1])))-1 ):
        output = residual_block(output)

    flat = Flatten()(output)
    output = Dense(target_.shape[1], activation='softmax')(flat)

    m = Model(input, output)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if loading == 'True':
        m.load_weights(path)
        print "Weights loaded!"
    #m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


# Record a WAV file
def audio2wav(output, fs):
    output_filename = './gen_audio.wav'
    output = output * 32768  # To convert range from [-1, 1] to [-32677, 32678] as a signed int
    data2 = np.asarray(output, dtype=np.int16)
    wavfile.write(filename=output_filename, rate=fs, data=data2)
    return
