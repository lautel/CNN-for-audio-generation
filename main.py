#!/usr/bin/env python
'''
Author: LAURA CABELLO PIQUERAS
Starting date: September 2016
Last update: 13/01/2017

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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from IPython import embed
from time import time
import config.nn_config as nn_config
from gen_utils.seed_generator import generate_seed_sequence
from gen_utils.sequence_generator import generate_from_seed
from data_utils.functions import *

# Data generation: several periods of 10000 sinusoid at different frecs
config = nn_config.get_parameters()

mode = config['mode_in']
b = config['nbits']
frequency = config['frequencies']
framerate = config['framerate']
seconds = config['sec']
batch_size = config['batch_size']
permut = config['permutations']

t = np.linspace(0, seconds, seconds*framerate)

# freq_set = list(itertools.permutations(frequency, r=3))[:permut]  # 1685040 permutations
# # divisor = int(np.floor(len(freq_set)/batch_size) * batch_size)
# # freq_set = freq_set[:divisor]
# freq_set = np.reshape(freq_set, (len(freq_set)/batch_size, batch_size, 3))
# phase = np.random.uniform(-180, 180, len(frequency))
# phase_set = list(itertools.permutations(phase, r=3))[:permut]  # [:divisor]
# phase_set = np.reshape(phase_set, (len(phase_set)/batch_size, batch_size, 3))

phase_set = [0, np.pi/2, 0]
# Generation of input sub-dataset
wave = generate_input(frequency, phase_set, t, mode=mode)
# Quantization of the signal with 'b' bits
[X, values] = quantifier(waves=wave, Amax=1, nbits=b)
# Redefinition of input and target data: predict the last sample one-hot encoded
data = one_hot_encoding(X, values)

plt.figure()
plt.plot(wave[0][:256])
plt.plot(X[0][:256], 'r')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.imshow(data[0][:500,:], interpolation='nearest', cmap=plt.cm.gray)
plt.show()


w = 256
data_train = data[:,:w,:]
target_train = data[:,(w+1),:]


## Network parameters ##
nb_filters = config['nb_filters']
nb_epoch = config['nb_epoch']
filepath = config['wfilepath']
nbatch = config['nbatch_per_epoch']
samples_per_epoch = batch_size*nbatch

# Callbacks
pb = printbatch()

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


np.save('acc_100_mix_256_9.npy', hist.history['acc'])
np.save('loss_100_mix_256_9.npy', hist.history['loss'])
np.save('acc_val_100_mix_256_9.npy', hist.history['val_acc'])
np.save('loss_val_100_mix_256_9.npy', hist.history['val_loss'])

embed()

acc = hist.history['acc']
loss = hist.history['loss']
acc_val = hist.history['val_acc']
loss_val = hist.history['val_loss']

# acc = np.load('acc_80_mix2_256_8.npy')
# loss = np.load('loss_80_mix2_256_8.npy')
# acc_val = np.load('acc_val_80_mix2_256_8.npy')
# loss_val = np.load('loss_val_80_mix2_256_8.npy')

plt.figure(1)
plt.plot(acc)
plt.plot(acc_val,'g')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')

# summarize history for loss in both training and validation datasets
plt.figure(2)
plt.plot(loss)
plt.plot(loss_val,'g')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

### TEST DATA ###
# Amount of samples to generate
samples = 50

# # Generation of test input
wavet = generate_input(100, phase_set, t, mode=['mixture2', 'mixture3'])
# Quantization of the signal with 'b' bits
[Xt, valuest] = quantifier(waves=wavet, Amax=1, nbits=b)
# Redefinition of input and target data: predict the last sample one-hot encoded
datat = one_hot_encoding(Xt, valuest)

sig=0
my_gen_test = my_generator_test(datat[sig], w, t)

### ---->>> NEED TO REVIEW THIS PART... SWITCH TO model.predict() INSTEAD
prediction = model.predict_generator(my_gen_test, val_samples=samples, max_q_size=10, nb_worker=1, pickle_safe=False)
predict_class = np.argmax(prediction, axis=1)
predict_value = values[predict_class]

oh_prediction = np.zeros((prediction.shape), dtype=int)
for j in range(oh_prediction.shape[0]):
    oh_prediction[j, predict_class[j]] = -1  # Asi se ven en rojo los puntos predecidos

### ERROR MEASURE ###
mae = mean_absolute_error(Xt[sig][w+1:w+samples+1], predict_value)
mse = mean_squared_error(Xt[sig][w+1:w+samples+1], predict_value)
print "MAE = " + str(mae)
print "MSE = " + str(mse)

# print('Evaluate: [loss, accuracy]', model.evaluate_generator(my_gen_test, val_samples=samples))

### REAL-VALUES VISUALIZATION ###
plt.figure(3)
plt.plot(t[:w+samples+1], Xt[sig][:w+samples+1])
plt.plot(t[w+1:w+1+samples], predict_value,'r')
plt.title('Sine generation after %i epochs training' % nb_epoch)
plt.legend(['Original', 'Generated'], loc='lower right')
plt.show()

### ONE-HOT VISUALIZATION ###
# total = np.concatenate((datat[0][:w,:], oh_prediction), axis=0)
# fig = plt.figure(4)
# ax = fig.add_subplot(1, 1, 1)
# ax.set_aspect('equal')
# plt.imshow(total, interpolation='nearest', cmap=plt.cm.RdBu)
# plt.show()


### SYNTHESIZE
if config['synthesize'] == 'True':
#
#     print ('\nStarting generation!\n')
#     #seed_len = 1
#     #seed_seq = generate_seed_sequence(seed_length=seed_len, training_data=data_train)
    [sequence, oneHseq] = generate_from_seed(model=model, seed=datat[sig][:w,:], sequence_length=samples,
                                             data_variance=np.var(frequency),data_mean=np.mean(frequency), values_=values)

    print ('Finished generation!')
#
    axe = np.arange(sequence.shape[1]) / sequence.shape[1]

    plt.figure(5)
    plt.plot(t[:w+1], sequence[0, :w+1], 'b')
    plt.plot(t[w+1:w+1+samples], sequence[0, w:], 'r')
    plt.show()


embed()

# #Save the generated sequence to a WAV file
fs = config['sampling_frequency']
audio2wav(predict_value, 16000)

