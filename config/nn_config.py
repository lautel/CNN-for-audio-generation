from __future__ import division
import numpy as np


def get_parameters():
    nn_params = {}
    nn_params['mode_in'] = ['mixture3']   # simple, mixture1, mixture2, mixture3
    nn_params['nbits'] = 8
    nn_params['framerate'] = 16e3
    nn_params['nfrec'] = 1
    #nn_params['frequencies'] = np.random.uniform(1, nn_params['framerate']/24, nn_params['nfrec'])
    nn_params['sec'] = 5

    nn_params['nb_filters'] = 32
    nn_params['batch_size'] = 256
    nn_params['nbatch_per_epoch'] = 12
    nn_params['nb_epoch'] = 80
    nn_params['permutations'] = nn_params['batch_size'] * nn_params['nbatch_per_epoch'] * nn_params['nb_epoch']
    nn_params['load_weight'] = 'False'
    nn_params['save_weight'] = 'True'
    nn_params['wfilepath'] = './weights/weights-simple-100e-256batch.hdf5'

    nn_params['synthesize'] = 'False'
    nn_params['generation-length'] = 2*(nn_params['framerate']-1)
    nn_params['sampling_frequency'] = 32000

    return nn_params