from __future__ import print_function
import numpy as np
from data_utils.functions import one_hot_encoding
#import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
def generate_rand_ind_batch(nb_samples, size_block, random=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(nb_samples)
    for i in range((nb_samples + (-nb_samples % size_block)) / size_block):  # roundup division
        indices = ind[i * size_block: (i + 1) * size_block]
        if len(indices)<size_block:
            indices = np.concatenate((indices, ind[:size_block-len(indices)]))
        yield indices


# ----------------------------------------------------------------------------------------------------------------------
def generate_epoch_train_test(data, _, window_length, batch_size, stride, random=True, debug=False):
    n_samples_epoch, wave_size = data.shape
    for ind_waves in generate_rand_ind_batch(n_samples_epoch, batch_size, random):
        offset = np.random.randint(0, wave_size - window_length - stride, n_samples_epoch)
        n_samples_batch = len(ind_waves)
        x = np.empty((n_samples_batch,window_length))
        y = np.empty((n_samples_batch, window_length))
        for i in range(n_samples_batch):
            if debug:
                print('... minibatch frame %d/%d, file: %d, offset: %d' % (i, n_samples_batch, ind_waves[i], offset[i]))
            x[i,:] = data[ind_waves[i], offset[i]:offset[i]+window_length]
            y[i,:] = data[ind_waves[i], offset[i]+stride:offset[i]+stride+window_length]
        yield x, y


# ----------------------------------------------------------------------------------------------------------------------
def generate_train_test(*args, **kwargs):
    while 1:
        #print("Starting epoch...")
        for x, y in generate_epoch_train_test(*args,**kwargs):
            x_oh = one_hot_encoding(x, args[1])
            y_oh = one_hot_encoding(y, args[1])
            yield x_oh, y_oh





# ----------------------------------------------------------------------------------------------------------------------
def generate_rand_ind_batch_v(nb_samples, size_block, random=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(nb_samples)
    for i in range((nb_samples + (-nb_samples % size_block)) / size_block):  # roundup division
        indices = ind[i * size_block: (i + 1) * size_block]
        if len(indices) < size_block:
            indices = np.concatenate((indices, ind[:size_block - len(indices)]))
        yield indices

# ----------------------------------------------------------------------------------------------------------------------
def generate_epoch_train_test_v(data, _, window_length, batch_size, stride, random=True, debug=False):
    n_samples_epoch, wave_size = data.shape
    for ind_waves in generate_rand_ind_batch_v(n_samples_epoch, batch_size, random):
        offset = np.random.randint(0, wave_size - window_length - stride, n_samples_epoch)
        n_samples_batch = len(ind_waves)
        x = np.empty((n_samples_batch, window_length))
        y = np.empty((n_samples_batch, window_length))
        for i in range(n_samples_batch):
            if debug:
                print('... minibatch frame %d/%d, file: %d, offset: %d' % (
                i, n_samples_batch, ind_waves[i], offset[i]))
            x[i, :] = data[ind_waves[i], offset[i]:offset[i] + window_length]
            y[i, :] = data[ind_waves[i], offset[i] + stride:offset[i] + stride+window_length]
        yield x, y

# ----------------------------------------------------------------------------------------------------------------------
def generate_valid_test(*args, **kwargs):
    while 1:
        # print("Starting epoch...")
        for x, y in generate_epoch_train_test_v(*args, **kwargs):
            x_oh = one_hot_encoding(x, args[1])
            y_oh = one_hot_encoding(y, args[1])
            yield x_oh, y_oh


# ----------------------------------------------------------------------------------------------------------------------
# test
if __name__ == "__main__":

    np.random.seed(1337)  # for reproducibility

    # ---------------------------------------
    # test simple random generator
    print("generating 15 / 5, (deterministic)")
    for x in generate_rand_ind_batch(15, 5, random=False):
        print(x)

    print("generating 15 / 5")
    for x in generate_rand_ind_batch(15, 5):
        print(x)

    print("generating 16 / 5")
    for x in generate_rand_ind_batch(16, 5):
        print(x)
