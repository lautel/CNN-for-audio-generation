#!/usr/bin/env python
'''
Author: LAURA CABELLO PIQUERAS
Starting date: September 2016
Last update: 14/01/2017

CNN with X hidden layers, where X depends on the length of the input.
Input to the network is a set of waves ONE-HOT encoded. I work with 3 sort of waves: simple sines, mixture of 2 or
sinusoids with different frequencies (and phases).
Scaling up the previous version, this waves are generated as larger sequences which will be feed to network in batches
of fixed length segments. This segment of length N, start out at the beginning of the signal so the sample N+1 is
predicted. Then, it is slided 1 sample (from sample 1 to N+1) so N+2 is predicted... and so on.

After training the network, we save and load the weights for generating a new wave [STILL WORKING ON IT]

'''

from __future__ import division
import numpy as np
import itertools
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from IPython import embed
from time import time
#from tqdm import tqdm
import config.nn_config as nn_config
from gen_utils.seed_generator import generate_seed_sequence
from gen_utils.sequence_generator import generate_from_seed
from data_utils.functions import *

# Data generation: several periods of 10000 sinusoid at different frecs
config = nn_config.get_parameters()

mode = config['mode_in']
b = config['nbits']
frequency = config['frequencies']
phase_set = config['phases']
framerate = config['framerate']
seconds = config['sec']
nwaves = config['nwaves']

t = np.linspace(0, seconds, seconds*framerate)


# Generation of input sub-dataset
wave = generate_input(nwaves, frequency, phase_set, t, mode=mode)
# Quantization of the signal with 'b' bits
[X, values] = quantifier(waves=wave, Amax=1, nbits=b)
# Redefinition of input and target data: predict the last sample one-hot encoded
data = one_hot_encoding(X, values)

# plt.figure()
# plt.plot(X[0][:240])
# plt.plot(X[1][:240])
# plt.plot(X[2][:240])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.set_aspect('equal')
# plt.imshow(data[0][:500,:], interpolation='nearest', cmap=plt.cm.gray)
# plt.show()

## define the checkpoint
pb = printbatch()
#filepath="./weights/weights-loss-{loss:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]


## Network parameters ##
nb_filters = config['nb_filters']
nb_epoch = config['nb_epoch']
filepath = config['wfilepath']
batch_size = config['batch_size']
nbatch = config['batches_per_epoch']

samples_per_epoch = batch_size * nbatch * nwaves

## Defining data input shape ##
w = 513
data_train = data[:,:w,:]
target_train = data[:,(w+1),:]

##### Defining the neural network ####
print 'Building neural network architecture...'

if config['load_weight'] == 'True':
    model = complex_model(data_train, target_train, nb_filters, framerate, loading='True', path=filepath)
    model.summary()
else:
    model = complex_model(data_train, target_train, nb_filters, framerate)
    model.summary()

    print "Training the CNN network..."
    init = time()
    my_gen = my_generator(data, w, t)
    my_gen_valid = my_generator_valid(data, w, t)
    hist = model.fit_generator(
        my_gen,
        samples_per_epoch=samples_per_epoch,  # number of examples I expect to see in an epoch
        nb_epoch=nb_epoch,
        validation_data=my_gen_valid,
        nb_val_samples=batch_size,
        class_weight=None,
        nb_worker=1
        #callbacks=[pb]
    )
    endt = time()
    elapsed_time = endt - init
    print("Elapsed time for training: %.10f seconds" % elapsed_time)

    if config['save_weight'] == 'True':
        model.save_weights(filepath)
        print('Weights saved!')


np.save('acc_80_mix_256_9_m.npy', hist.history['acc'])
np.save('loss_80_mix_256_9_m.npy', hist.history['loss'])
np.save('acc_val_80_mix_256_9_m.npy', hist.history['val_acc'])
np.save('loss_val_80_mix_256_9_m.npy', hist.history['val_loss'])


### TEST DATA ###

### SYNTHESIZE
if config['synthesize'] == 'True':

    acc = np.load('acc_80_mix_256_9_m.npy')
    loss = np.load('loss_80_mix_256_9_m.npy')
    acc_val = np.load('acc_val_80_mix_256_9_m.npy')
    loss_val = np.load('loss_val_80_mix_256_9_m.npy')

    plt.figure(1)
    plt.plot(acc)
    plt.plot(acc_val,'g')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')

    # summarize history for loss in both training and validation datasets
    plt.figure(2)
    plt.plot(loss)
    plt.plot(loss_val,'g')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    print ('Starting generation!\n')

    samples = config['samples']
    sig = 10
    # # Generation of test input
    # wavet = generate_input(2,[110,200], phase_set, t)
    # Quantization of the signal with 'b' bits
    # [Xt, valuest] = quantifier(waves=wavet, Amax=1, nbits=b)
    # Redefinition of input and target data: predict the last sample one-hot encoded
    # datat = one_hot_encoding(Xt, valuest)
    Xt = X
    seedSeq = data[sig, :w, :]
    seedSeq = np.reshape(seedSeq, (1, seedSeq.shape[0], seedSeq.shape[1]))
    # seed_len = 1
    # seed_seq = generate_seed_sequence(seed_length=seed_len, training_data=data_train)
    [sequence, oneHseq] = generate_from_seed(model=model, seed=seedSeq, sequence_length=samples,
                                             data_variance=np.var(frequency), data_mean=np.mean(frequency),
                                             values_=values)
    prediction = sequence[0, w:]
    print ('Finished generation!')

    ### ERROR MEASURE ###
    mae = mean_absolute_error(Xt[sig][w+1:w+samples+1], prediction)
    mse = mean_squared_error(Xt[sig][w+1:w+samples+1], prediction)
    print "MAE = " + str(mae)
    print "MSE = " + str(mse)

    ### REAL-VALUES VISUALIZATION ###
    plt.figure(3)
    plt.plot(t[:w+1], sequence[0, :w+1], 'b')
    plt.plot(t[w+1:w+1+samples], sequence[0, w:], 'r')
    plt.title('Sine generation after %i epochs training' % nb_epoch)
    plt.legend(['Original', 'Generated'], loc='lower right')
    plt.show()

    ### ONE-HOT VISUALIZATION ###
    # fig = plt.figure(4)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_aspect('equal')
    # plt.imshow(oneHseq, interpolation='nearest', cmap=plt.cm.RdBu)
    # plt.show()

# print('Evaluate: [loss, accuracy]', model.evaluate_generator(my_gen_test, val_samples=samples))

embed()


# #Save the generated sequence to a WAV file
# fs = config['sampling_frequency']  #--> tiene que ser la misma Fs con la que se generan los senos o es independiente?
# audio2wav(predict_value, 16000)

