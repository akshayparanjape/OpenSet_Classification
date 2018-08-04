'''
CIFAR 10 dataset
dataset is storeed locally

'''


import os
import tensorflow as tf

IMAGE_SIZE = 32  # 32 is max
NUM_CLASSES = 10  # 10 is max
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000  # 50000 is max
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = min(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                                      10000)  # 10000 is max

DATA_DIR = './cifar10_data'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def get_dataset(eval_data, batch_size, num_epochs, perform_shuffle):
    """Return a Dataset which will feed the network with data (images and
    labels).

    Converts the data from binary format to images and labels. Applies
    preprocessing and sets up the Dataset to return batches for a fixed
    number of epochs.
    """

    data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Check that all files exist
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    label_bytes = 1  # 2 for CIFAR-100
    height = 32  # original format
    width = 32
    depth = 3  # RGB
    image_bytes = height * width * depth
    # Every record consists of a label followed by the image, with a fixed
    # number of bytes for each
    record_bytes = label_bytes + image_bytes

    # The Cifar10 dataset is stored in binary format
    cifar_dataset = tf.data.FixedLengthRecordDataset(filenames,
                                                     record_bytes=record_bytes)

    def decode_bin_data(line):
        # Convert from a string to a vector of unit8 that is record_bytes long
        line = tf.decode_raw(line, tf.uint8)

        # The first bytes is the label. We slice  it and convert to int32
        label = tf.cast(tf.strided_slice(line, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we
        # reshape from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.strided_slice(line, [label_bytes],
                                 [label_bytes + image_bytes]),
                                 [depth, height, width])
        # Convert from [depth, height, width] to [height, width, depth].
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        image = tf.cast(uint8image, tf.float32)
        image.set_shape([height, width, 3])
        label.set_shape([1])
        return image, label

    # Decode the images from binary
    cifar_dataset = cifar_dataset.map(decode_bin_data)
    

    # Only use a sub-set of the data to speed things up
    cifar_dataset = cifar_dataset.take(num_examples_per_epoch)

    # Apply preprocessing: Crop, distort etc.

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        cifar_dataset = cifar_dataset.shuffle(4 * batch_size)

	
#	cifar_dataset = cifar_dataset.batch(batch_size)
#	cifar_dataset = cifar_dataset.repeat(num_epochs)
	return cifar_dataset
	

# Shuffling so that in every epoch there is different order for images, may be to avoid over fitting
print(get_dataset(0,10,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,1))