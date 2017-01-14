from __future__ import division
import numpy as np


def get_parameters():
    nn_params = {}
    nn_params['mode_in'] = 'simple'
    nn_params['nbits'] = 8
    nn_params['framerate'] = 16e3
    nn_params['nwaves'] = 20
    nn_params['frequencies'] = np.random.uniform(1, nn_params['framerate']/24, nn_params['nwaves'])
    nn_params['phases'] = [0, np.pi/2, np.pi]
    nn_params['sec'] = 1

    nn_params['nb_filters'] = 32
    nn_params['batch_size'] = 256
    nn_params['batches_per_epoch'] = 10
    nn_params['nb_epoch'] = 80
    #nn_params['permutations'] = nn_params['batch_size'] * nn_params['nbatch_per_epoch'] * nn_params['nb_epoch']
    nn_params['load_weight'] = 'False'
    nn_params['save_weight'] = 'True'
    nn_params['wfilepath'] = './weights/weights-simple-80e-256batch.hdf5'

    nn_params['synthesize'] = 'True'
    nn_params['samples'] = 50

    return nn_params
