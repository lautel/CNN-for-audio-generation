from __future__ import division
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Convolution1D, Convolution2D, Merge, Activation, \
    TimeDistributed, Reshape, Permute, AtrousConvolution1D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
from keras.regularizers import l2, activity_l2
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.engine import Model
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
    #     if batch%10 == 0:
    #         print "\nBatch " + str(batch) + " ends"

    def on_epoch_begin(self, epoch, logs={}):
        print(logs)

    def on_epoch_end(self, epoch, logs={}):
        print(logs)



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
    if loading == 'True':
        m.load_weights(path)
        print "Weights loaded!"
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m




# Architecture #3 for the CNN
def complex_model(output_levels, segment_length, receptive_field, nb_filters_, framerate_, loading=False, path=""):
    fnn_init = 'he_uniform'
    def residual_block(input_):
        original = input_
        tanh_ = AtrousConvolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            atrous_rate=2**i,
            init=fnn_init,
            border_mode='valid',
            bias=False,
            causal=True,
            activation='tanh',
            name='AtrousConv1D_%d_tanh' % (2**i)
        )(input_)

        sigmoid_ = AtrousConvolution1D(
            nb_filter=nb_filters_,
            filter_length=2,
            atrous_rate=2**i,
            init=fnn_init,
            border_mode='valid',
            bias=False,
            causal=True,
            activation='sigmoid',
            name='AtrousConv1D_%d_sigm' % (2**i)
        )(input_)

        input_ = Merge(mode='mul')([tanh_, sigmoid_])

        res_x = Convolution1D(nb_filter=nb_filters_, filter_length=1, border_mode='same', bias=False)(input_)
        skip_c = res_x
        res_x = Merge(mode='sum')([original, res_x])

        return res_x, skip_c

    input = Input(shape=(segment_length, output_levels), name='input_part')
    skip_connections = []
    output = input
    output = AtrousConvolution1D(
        nb_filter=nb_filters_,
        filter_length=2,
        atrous_rate=1,
        init=fnn_init,
        activation='relu',
        border_mode='valid',
        causal=True,
        name='initial_AtrousConv1D'
    )(output)

    for i in range( int(np.log2( receptive_field ) ) ):
        output, skip_c = residual_block(output)
        skip_connections.append(skip_c)

    out = Merge(mode='sum')(skip_connections)

    for _ in range(2):
        out = Activation('relu')(out)
        out = Convolution1D(output_levels, 1, activation=None, border_mode='same')(out)
    out = Activation('softmax', name='output_softmax')(out)

    m = Model(input, out)
    if loading:
        m.load_weights(path)
        print "Weights loaded!"
    #ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, decay=0.0)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m





# Architecture #4 for the CNN
def complex_model2(data_, target_, nb_filters_, framerate_, loading=False, path=""):
    fnn_init = 'he_uniform'
    def residual_block(input_):

        tanh_ = Convolution2D(
            nb_filter=nb_filters_,
            nb_row=2,
            nb_col=2,
            subsample=(2,2),
            dim_ordering='th',
            init=fnn_init,
            activation='tanh'
        )(input_)

        sigmoid_ = Convolution2D(
            nb_filter=nb_filters_,
            nb_row=2,
            nb_col=2,
            subsample=(2,2),
            dim_ordering='th',
            init=fnn_init,
            activation='sigmoid'
        )(input_)

        out = Merge(mode='mul')([tanh_, sigmoid_])

        return out

    input = Input(shape=(data_.shape[1], data_.shape[2], data_.shape[3]), name='input_part')
    output = Convolution2D(
        nb_filter=nb_filters_,
        nb_row=2,
        nb_col=2,
        dim_ordering='th',
        init=fnn_init,
        activation='relu',
        border_mode='same'
    )(input)

    ply = int(np.log2( data_.shape[3]))-1
    for _ in range( ply ):
        output = residual_block(output)


    out = Permute((2, 1, 3))(output)
    dim1 = int(data_.shape[2]/(2**ply))
    dim2 = int(data_.shape[3]/(2**ply))

    out = Reshape((dim1, nb_filters_*dim2))(out)
    output = TimeDistributed(Dense(output_dim=target_.shape[2], init=fnn_init, activation='softmax'))(out)

    m = Model(input, output)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if loading:
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
