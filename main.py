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

from __future__ import division, print_function
#import theano
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.io import wavfile
from keras.models import Sequential
from IPython import embed
from time import time
#from tqdm import tqdm
import config.nn_config as nn_config
from gen_utils.seed_generator import generate_seed_sequence
from gen_utils.sequence_generator import generate_from_seed
from data_utils.functions import *
from generator import generate_train_test, generate_valid_test

#[data, samplerate] = np.fromfile('aa.raw', dtype=np.uint16)
#plt.plot(data[:100])
#plt.show()

#embed()
#theano.Mode(optimizer='fast_compile')
# Data generation: several periods of 10000 sinusoid at different frecs
config = nn_config.get_parameters()

mode = config['mode_in']
b = config['nbits']
frequency = config['frequencies']
framerate = config['framerate']
seconds = config['sec']
nwaves = config['nwaves']
output_levels = 2**b


#t = np.linspace(0, seconds, (framerate*(1/seconds)))
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
wave = generate_input(nwaves, frequency, phase_set, t, mode=mode)
# Quantization of the signal with 'b' bits
[X, values] = quantifier(waves=wave, Amax=1, nbits=b)
X = np.array(X)

## define the checkpoint
pb = printbatch()
#filepath="./weights/weights-loss-{loss:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

## Defining data input ##
# valid = int(nwaves/4)
# data_train = data[:valid, :, :]
# data_valid = data[-valid:, :,:]
# data_test = data[0][-nsignalv5 ECTS credits are reasonable:,:]
# target_train = data[0,w:,:]
# target_valid = data[1,w:,:]
# target_test = target_mat[-nsignalv:]

## Network parameters ##
receptive_field = config['receptive_field']
stride = config['stride']
window_length = receptive_field

nb_filters = config['nb_filters']
nb_epoch = config['nb_epoch']
filepath = config['wfilepath']
batch_size = config['batch_size']
nbatch = config['batches_per_epoch']

x_train = X[:-int(X.shape[0]/4):,:]
x_valid = X[-int(X.shape[0]/4):,:]
####
#samples_train = int((nwaves - valid) * len(xrange(0, data.shape[1] - window_length, 128)) / batch_size) * batch_size
#samples_valid = int((nwaves - valid) * len(xrange(0, data.shape[1] - window_length, 128)) / batch_size) * batch_size
samples_per_epoch = batch_size * nbatch
samples_per_epoch_v = int(batch_size/2) * nbatch
###

# print ("-- possible batches: ", freq_set.shape[0])

##### Defining the neural network ####
print("Building neural network architecture...")
if config['load_weight'] == 'True':
    model = complex_model(output_levels, window_length, receptive_field, nb_filters, framerate, loading=True, path=filepath)
    model.summary()
else:
    model = complex_model(output_levels, window_length, receptive_field, nb_filters, framerate)
    model.summary()

    print("Training the CNN network...")
    init = time()
    data_generator = generate_train_test(x_train, values, window_length, batch_size, stride, random=True)
    data_valid_generator = generate_valid_test(x_valid, values, window_length, int(batch_size/2), stride, random=True)

    hist = model.fit_generator(data_generator,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            validation_data=data_valid_generator,
                            nb_val_samples=samples_per_epoch_v,
                            #callbacks=[pb],
                            verbose=1)

    endt = time()
    elapsed_time = endt - init
    print("Elapsed time for training: %.10f seconds" % elapsed_time)

    if config['save_weight'] == 'True':
        model.save_weights(filepath)
        print('Weights saved!')


#np.save('acc_segments_5.npy', hist.history['acc'])
#np.save('loss_segments_5.npy', hist.history['loss'])
#np.save('acc_val_segments_5.npy', hist.history['val_acc'])
#np.save('loss_val_segments_5.npy', hist.history['val_loss'])

# acc = hist.history['acc']
# loss = hist.history['loss']
# acc_val = hist.history['val_acc']
# loss_val = hist.history['val_loss']


### TEST DATA ###

### SYNTHESIZE
if config['synthesize'] == 'True':
    # acc = np.load('acc_80_mix_256_9_m.npy')
    # loss = np.load('loss_80_mix_256_9_m.npy')
    # acc_val = np.load('acc_val_80_mix_256_9_m.npy')
    # loss_val = np.load('loss_val_80_mix_256_9_m.npy')
    #
    # plt.figure(1)
    # plt.plot(acc)
    # plt.plot(acc_val,'g')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    #
    # # summarize history for loss in both training and validation datasets
    # plt.figure(2)
    # plt.plot(loss)
    # plt.plot(loss_val,'g')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    print ('Starting generation!\n')


    # def test_preprocess(desired_sample_rate, batch_size, use_ulaw):
    #     sample_dir = os.path.join('preprocess_test')
    #     if not os.path.exists(sample_dir):
    #         os.mkdir(sample_dir)
    #
    #     ulaw_str = '_ulaw' if use_ulaw else ''
    #     sample_filename = os.path.join(sample_dir, 'test1%s.wav' % ulaw_str)
    #     sample_stream = make_sample_stream(desired_sample_rate, sample_filename)
    #
    #     outputs = generate_train_test(x_train, values, window_length, batch_size, stride, random=True).next()[0][1].astype('uint8')
    #
    #     write_samples(sample_stream, outputs)
    #     wavfile.write(os.path.join(sample_dir, 'test2%s.wav' % ulaw_str), desired_sample_rate,
    #                            np.argmax(outputs, axis=-1).astype('uint8'))
    #
    #
    # def make_sample_stream(desired_sample_rate, sample_filename):
    #     sample_file = wave.open(sample_filename, mode='w')
    #     sample_file.setnchannels(1)
    #     sample_file.setframerate(desired_sample_rate)
    #     sample_file.setsampwidth(1)
    #     return sample_file
    #
    #
    # def write_samples(sample_file, out_val, use_ulaw):
    #     s = np.argmax(out_val, axis=-1).astype('uint8')
    #     # print s,
    #     s = bytearray(list(s))
    #     # print s[0]
    #     sample_file.writeframes(s)
    #     sample_file._file.flush()


    samples = config['samples']
    sig = 0
    # # Generation of test input
    # wavet = generate_input(2,[110,200], phase_set, t)
    # Quantization of the signal with 'b' bits
    # [Xt, valuest] = quantifier(waves=wavet, Amax=1, nbits=b)
    # Redefinition of input and target data: predict the last sample one-hot encoded
    # datat = one_hot_encoding(Xt, valuest)
    Xt = X
    seedSeq = X[sig, :window_length, :]
    seedSeq = np.reshape(seedSeq, (1, seedSeq.shape[0], seedSeq.shape[1]))
    # seed_len = 1
    # seed_seq = generate_seed_sequence(seed_length=seed_len, training_data=data_train)
    [sequence, oneHseq] = generate_from_seed(model=model, seed=seedSeq, sequence_length=samples,
                                             data_variance=np.var(frequency), data_mean=np.mean(frequency),
                                             values_=values)
    prediction = sequence[0, window_length:]
    print('Finished generation!')

    ### ERROR MEASURE ###
    mae = mean_absolute_error(Xt[sig][window_length+1:window_length+samples+1], prediction)
    mse = mean_squared_error(Xt[sig][window_length+1:window_length+samples+1], prediction)
    print("MAE = " + str(mae))
    print("MSE = " + str(mse))

    ### REAL-VALUES VISUALIZATION ###
    # plt.figure(3)
    # plt.plot(t[:window_length+1], sequence[0, :window_length+1], 'b')
    # plt.plot(t[window_length+1:window_length+1+samples], sequence[0, window_length:], 'r')
    # plt.title('Sine generation after %i epochs training' % nb_epoch)
    # plt.legend(['Original', 'Generated'], loc='lower right')
    # plt.show()
    #
    # ## ONE-HOT VISUALIZATION ###
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

