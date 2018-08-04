'''
This file includes functions for training models
This contains the main function
'''

import tensorflow as tf
import numpy as np

import dataset
import preprocess as cifar10

# Batch size is number of examples trained at a time

NUM_STEPS_PER_EPOCH = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//cifar10.BATCH_SIZE + 1 
			# +1 since it may not be exactly divisable

# TODO: where is it used and why is it this number
RANDOM_SEED = 9564365

# FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_steps_per_epoch', NUM_STEPS_PER_EPOCH,
                            """Number of batches to run.""")			

# TODO: Complete evaluate
'''
def evaluate(sees,) :

	true_count = 0
	num_eval_samples = cifar10_data.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	num_steps = num_eval_samples // FLAGS.batch_size

'''


def train():
	return



# TODO: write main function for training and evaluating the datset
