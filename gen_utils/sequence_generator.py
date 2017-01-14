import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

# Based on this resource: https://github.com/MattVitelli/GRUV/blob/master/gen_utils/sequence_generator.py


def generate_from_seed(model, seed, sequence_length, data_variance, data_mean, values_):
    #seed = np.reshape(seed, (1,seed.shape[0], seed.shape[1]))
    seedSeq = seed.copy()
    output=[]
    seedNum = values_[np.argmax(seedSeq[0], axis=1)]
    seedNum = np.reshape(seedNum, (1, len(seedNum)))
    out = np.concatenate((seedNum, np.zeros((1,sequence_length))), axis=1)

    for it in xrange(sequence_length):
        prediction = model.predict(seedSeq)  # Step 1. Generate sample X_n + 1
        predict_class = np.argmax(prediction, axis=1)
        out[0,it+seedNum.shape[1]] = values_[predict_class]

        seedSeqNew = np.zeros((1,seedSeq.shape[2]))
        seedSeqNew[0, predict_class] = 1
        # one-hot encoding for making the prediction again

        if it == 0:
            output.append(np.concatenate((seedSeq[0], seedSeqNew.copy()), axis=0))
        else:
            output.append(seedSeqNew[0][0:seedSeqNew.shape[1]].copy())
        newSeq = np.reshape(seedSeqNew, (1, 1, seedSeqNew.shape[1]))
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
        seedSeq = np.delete(seedSeq, 0, axis=1)

    return out, output
