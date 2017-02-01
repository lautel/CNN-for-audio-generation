from __future__ import division
import numpy as np


def get_parameters():
    nn_params = {}
    nn_params['mode_in'] = 'mixture1'
    nn_params['nbits'] = 8
    nn_params['framerate'] = 8e3  ##
    nn_params['nwaves'] = 100
    nn_params['frequencies'] = np.random.uniform(1, nn_params['framerate']/24, nn_params['nwaves']+1)
    nn_params['phases'] = [0, np.pi/2, np.pi]
    nn_params['sec'] = 1

    nn_params['receptive_field'] = 1024
    nn_params['stride'] = 1 #
    nn_params['nb_filters'] = 256
    nn_params['batch_size'] = 32
    nn_params['batches_per_epoch'] = 10
    nn_params['nb_epoch'] = 50
    nn_params['load_weight'] = True
    nn_params['save_weight'] = False
    nn_params['wfilepath'] = './weights/weights-mix1-segments_3.hdf5'

    nn_params['synthesize'] = True
    nn_params['samples'] = 2000

    return nn_params



