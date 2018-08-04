import tensorflow as tf



# Function to get variables for weight for neurons

def _variable_on_cpu():
	#a = tf.device('/cpu:0')
	tf.get_variable('var_check', [12,23],tf.constant_initializer(0.0) , tf.float32)
	return 0


#print(_variable_on_cpu())

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a random Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    # Varibales on cpu used for bias
    var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	# initialized variable with weigft decay
	# wd is weight decay
	var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

	return var

# print(_variable_with_weight_decay('bias',[20],1,None) )



def linear_layer(name, x, shape, stddev=0.04, bias_init=0.0, wd=None):
    """
    A linear network layer (identity activation function)
    :param name: (str) The name of the layer, e.g. 'linear_layer'
    :param x:  The input to the layer
    :param shape: The shape of the weight matrix, e.g. [x_dim, z_dim[
    :param stddev: (float) standard deviation used to initialize the weights
    :param bias_init: (float) constant value used to initialize the bias vec
    :param wd: (float) weight decay value (use None instead of 0)
    :return:
    """
    # The reuse=tf.AUTO_REUSE is needed to do training and
    # validation on the same graph
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=shape,
                                              stddev=stddev, wd=wd)
        biases = _variable_on_cpu('biases', [shape[-1]],
                                  tf.constant_initializer(bias_init))
        z = tf.matmul(x, weights) + biases
        _activation_summary(z)  # Adds summary of z to Tensorboard
    return z




def full_layer(name, x, shape, stddev=0.04, bias_init=0.0, wd=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        #z = x
        # ====================================================================
        # Add code for fully-connected layer, shape = [inp_dim, 4096]
        weights = _variable_with_weight_decay('weights',
                                              shape=shape,
                                              stddev=stddev, wd=wd)
        biases = _variable_on_cpu('biases', [shape[-1]],
                                  tf.constant_initializer(bias_init))
        z = tf.matmul(x, weights) + biases

        # Activation tanh
#        z = tf.nn.tanh(z,'tanh_activation')

        z = tf.nn.relu(z,'relu_activation')
        #Ativation relu
        #z = tf.nn.relu(z,'relu_activation')

        _activation_summary(z)
        # ====================================================================
    return z

def conv_layer(name, x, shape, strides, padding, stddev=0.04, bias_init=0.0,
               wd=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        #z = x
        # ====================================================================
        # Add code for convolutional layer
        # ====================================================================

    # TODO: check why do we do kernel/ weights in this manner
    #               https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    #               check the implementation of weights/kernel


    # shape should be of the form [3x3x64]

        kernel = _variable_with_weight_decay('weights',
                                              shape=shape,
                                              stddev=stddev, wd=wd)

        conv   = tf.nn.conv2d(x,kernel,strides,padding=padding)

        biases = _variable_on_cpu('biases', [shape[-1]],
                                  tf.constant_initializer(bias_init))
        
        # tf.nn.bias_add  it adds bias to the value
        pre_activation = tf.nn.bias_add(conv, biases)

        # final step - applying the activation function
        z  = tf.nn.relu(pre_activation, name=scope.name)

    return z

def conv_layer_norm(name, x, is_training, shape, strides, padding, stddev=0.04,
                    bias_init=0.0, wd=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # z = x
        # ====================================================================
        # Add code for convolutional layer
        # ====================================================================
        kernel = _variable_with_weight_decay('weights',
                                              shape=shape,
                                              stddev=stddev, wd=wd)

        conv   = tf.nn.conv2d(x,kernel,strides,padding=padding)

        biases = _variable_on_cpu('biases', [shape[-1]],
                                  tf.constant_initializer(bias_init))

        pre_activation = tf.nn.bias_add(conv, biases)
        norm = tf.layers.batch_normalization(pre_activation,
                                             training=is_training,
                                             name='norm')

        # final step - applying the activation function
        z  = tf.nn.relu(pre_activation, name=scope.name)
    return z

#===============================================
#  classes
#===============================================

class Cifar10Linear:
    def __init__(self, num_classes, wdecay_conv, wdecay_full):
        self.wd_conv = wdecay_conv
        self.wd_full = wdecay_full
        self.num_classes = num_classes

    def __call__(self, images, eval_data):
        # Reshape images to vectors
        shape_conv = images.get_shape().as_list()
        dim = shape_conv[1] * shape_conv[2] * shape_conv[3]
        reshape = tf.reshape(images, [-1, dim])
        # Apply linear layer to the reshaped images
        logits = linear_layer('full_linear', reshape, [dim, self.num_classes],
                              stddev=0.04, bias_init=0.0, wd=self.wd_full)
        return logits


class Cifar10_MLP:
    def __init__(self, num_hidden_neuron, num_classes, wdecay_conv, wdecay_full):
        self.hidden_layer_neurons = num_hidden_neuron        
        self.wd_full = wdecay_full
        self.num_classes = num_classes


    def __call__(self, images, eval_data):
        # Reshape images to vectors
        shape_conv = images.get_shape().as_list()
        dim = shape_conv[1] * shape_conv[2] * shape_conv[3]
        reshape = tf.reshape(images, [-1, dim])
        # Apply hidden layer to the reshaped images
        hidden = full_layer('MLP', reshape, [dim, self.hidden_layer_neurons],
                              stddev=0.04, bias_init=0.0)

        logits = linear_layer('full_linear', hidden, [self.hidden_layer_neurons, self.num_classes],
                              stddev=0.04, bias_init=0.0, wd=self.wd_full)

        return logits


class Cifar10_Conv_pool:
    def __init__(self, num_classes, wdecay_conv, wdecay_full):             
        self.num_classes = num_classes
        self.wd_full = wdecay_full       
        self.wd_conv = wdecay_conv

    def __call__(self,images,eval_data):
 
        conv1 = conv_layer('Conv1', images,ksize=[1,2,2,1],strides=[1,2,2,1], padding = 'SAME', stddev=0.04, bias_init=0.0,
               wd=None)

        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],
                         padding='SAME', name='pool1')

        conv2 = conv_layer('Conv2', pool1, [3,3,64,64],  strides=[1,2,2,1], padding = 'SAME', stddev=0.04, bias_init=0.1,
               wd=None)

        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],
                         padding='SAME', name='pool2')

       #shape_conv = images.get_shape().as_list()
        shape_conv =  pool2.get_shape().as_list();
        dim = shape_conv[1] * shape_conv[2] * shape_conv[3]
        reshape = tf.reshape(pool2, [-1, dim])

        logits = linear_layer('full_linear', reshape, [dim, self.num_classes],
                              stddev=0.04, bias_init=0.0, wd=self.wd_full)

        return logits


class Cifar10_Conv_pool_norm:
    def __init__(self, num_classes, wdecay_conv, wdecay_full):
        self.wd_conv = wdecay_conv
        self.wd_full = wdecay_full
        self.num_classes = num_classes

    def __call__(self, images, eval_data):
        conv1 = conv_layer_norm('conv1', images, is_training=not eval_data,
                                shape=[3, 3, 3, 64], strides=[1, 1, 1, 1],
                                padding='SAME', stddev=0.04, bias_init=0.0,
                                wd=self.wd_conv)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

        conv2 = conv_layer_norm('conv2', pool1, is_training=not eval_data,
                                shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                                padding='SAME', stddev=0.04, bias_init=0.1,
                                wd=self.wd_conv)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')


        shape_conv = pool2.get_shape().as_list()
        dim = shape_conv[1] * shape_conv[2] * shape_conv[3]
        reshape = tf.reshape(pool2, [-1, dim])
        logits = linear_layer('logits', reshape, [dim, self.num_classes],
                              stddev=0.04, bias_init=0.1,
                              wd=self.wd_full)

        return logits

# using larger strides
        