import numpy as np
from tqdm import tqdm
from IPython import embed
from data_utils.functions import one_hot_encoding

# Extrapolates from a given seed sequence
# Source: https://github.com/MattVitelli/GRUV/blob/master/gen_utils/sequence_generator.py


def generate_from_seed(model, seed, sequence_length, data_variance, data_mean, values_):
    #seed = np.reshape(seed, (1,seed.shape[0], seed.shape[1]))
    seedSeq = seed.copy()

    #seedNum = values_[np.argmax(seedSeq[0])]
    #seedNum = np.reshape(seedNum, (1, len(seedNum)))
    out = np.concatenate((seedSeq, np.zeros((1,sequence_length))), axis=1)
    #out=np.zeros((1,sequence_length))
    for it in tqdm(xrange(sequence_length)):
        seedSeq = out[:, it : it+seedSeq.shape[1]]
        oh_seed = one_hot_encoding(seedSeq, values_)
        prediction = model.predict(oh_seed, batch_size=32)  # Step 1. Generate sample X_n + 1
        predict_class = np.argmax(prediction[0], axis=1)
        out[0,it+seedSeq.shape[1]] = values_[predict_class[-1]]

    # Finally, post-process the generated sequence so that we have valid frequencies
    # We're essentially just undo-ing the data centering process
    # for i in xrange(len(output)):
    #     output[i] *= data_variance
    #     output[i] += data_mean
    return out
