'''

This file include some minor pre-processsin for the cifar 10 model and few functions

# Compute input images and labels for training. If you would like to run
 inputs, labels = inputs()

# Compute inference on the model inputs to make a prediction.
 logits = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(logits, labels)

 all this function will be calculated for the batch size

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
MODEL = 'Cifar10_Conv_pool'
MODEL_ARGS = (NUM_CLASSES, WEIGHT_DECAY_CONV, WEIGHT_DECAY_FULL)

# FLAGS available in module
# Flags available in all modules
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', NUM_EPOCHS,
                            """Number of training epochs.""")


# get input in the form  (input,image)
def input(eval_data, perform_shuffle):

	num_epochs = FLAGS.num_epochs if not eval_data else None
	data = dataset.get_dataset(eval_data,FLAGS.batch_size, num_epochs, perform_shuffle)
	
	# Create an iterator over the dataset
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    labels = tf.reshape(labels, [-1])  # Make sure labels is a 1-D tensor

	return images,labels


# prediction for the image according to the class, this will return the logits
def inference(images, eval_data):
	if hasattr(model,MODEL):
		ANN = getattr(model, MODEL)
	else:
		raise ValueError('Model %s not present in cifar10_model' % MODEL)
	ANN = ANN(*MODEL_ARGS) # construction of the class
	return ANN(images,eval_data)


# calculating the total loss with respect to the label, logits
'''
We can use mean squarred error
Cross entropy loss
'''
# Loss function
def loss(logits,labels):

	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name = 'cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
	return cross_entropy_mean



'''
cross entropy  mean  = 1/N * ()

'''


