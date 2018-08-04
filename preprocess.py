'''

This file include some minor pre-processsin for the cifar 10 model

'''


import tensorflow as tf
import dataset
import model

IMAGE_SIZE = dataset.IMAGE_SIZE
NUM_CLASSES = dataset.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# This constants can also be put in tf.FLAGS

# defining constants | HYPERPARAMETERS

LEARNING_RATE = 0.1               # Initial learning rate.
NUM_EPOCHS = 20               # Number of training epochs
WEIGHT_DECAY_CONV = None          # Value of weight decay for Conv layer
WEIGHT_DECAY_FULL = 0.004         # For fully connected layers (use None instead of 0)
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
BATCH_SIZE = 64

# TODO: what does weight decay do  ?

# Model Constants


# FLAGS available in module
# Flags available in all modules
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', NUM_EPOCHS,
                            """Number of training epochs.""")
