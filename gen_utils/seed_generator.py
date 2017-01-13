import numpy as np

# A very simple seed generator
# Copies a random example's first seed_length sequences as input to the generation algorithm
# Code based on this source: https://github.com/MattVitelli/GRUV/blob/master/gen_utils/seed_generator.py

def generate_seed_sequence(seed_length, training_data):
    num_examples = training_data.shape[0]
    randIdx = np.random.randint(num_examples, size=1)[0]
    # Working with tuples is faster than with lists (make a tuple = freeze a list)
    randSeed = np.concatenate(tuple([training_data[randIdx + i] for i in xrange(seed_length)]), axis=0)
    seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))

    return seedSeq