import tensorflow as tf
import numpy as np
import import_images
import pickle

# TODO - currently missing weight decay (?) and local response regularization

# parameters
hPix = 240
wPix = 376
num_classes = 3

learning_rate = 0.001
momentum = 0.09

batch_size = 32
training_iterations = 100
batch_increments = 128
strides = [1, 1, 1, 1]
ksize = [1, 3, 3, 1]  # size of the window for each dimension of the input tensor
maxpool_strides = [1, 2, 2, 1] # stride of the sliding window for each dimension of the input tensor

# placeholders for the probability that a neuron's output is kept during dropout (to prevent overfitting)
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")  # difference between keep_conv and keep_hidden?

# return a random tensor w/ normal dist as a variable
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# return convolutional neural network formulation
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    '''
    w = init_weights([3, 3, 1, 6])
    w2 = init_weights([3, 3, 6, 12])
    w3 = init_weights([3, 3, 12, 24])
    w4 = init_weights([24 * 4 * 4, 48])
    w_o = init_weights([48, 3])
    '''
    # First conv layer
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides, 'SAME'))
    print "l1a.shape:", l1a.get_shape()  # for debugging
    l1 = tf.nn.max_pool(l1a, ksize, maxpool_strides, padding='SAME')
    #l1 = tf.nn.dropout(l1, p_keep_conv)
    print "l1.shape:", l1.get_shape()

    # Second conv layer
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides, 'SAME'))
    print "l2a.shape:", l2a.get_shape()
    l2 = tf.nn.max_pool(l2a, ksize, maxpool_strides, padding='SAME')
    #l2 = tf.nn.dropout(l2, p_keep_conv)
    print "l2.shape:", l2.get_shape()

    # Third conv layer
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides, 'SAME'))
    print "l3a.shape:", l3a.get_shape()
    l3 = tf.nn.max_pool(l3a, ksize, maxpool_strides, padding='SAME')
    print "l3.shape:", l3.get_shape()

    # Fully connected layer 1
    l3Shape = l3.get_shape().as_list()
    print "list", l3Shape
    l3 = tf.reshape(l3, [-1, l3Shape[1] * l3Shape[2] * l3Shape[3]])      # problem area
    l3 = tf.nn.dropout(l3, p_keep_conv)
    print "l3.shape:", l3.get_shape()

    # Fully connected layer 2
    l4 = tf.nn.relu(tf.matmul(l3, w4) + bf1)
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    print "l4.shape:", l4.get_shape()

    # Output layer
    pyx = tf.matmul(l4, w_o) + bf2
    print "pyx.shape:", pyx.get_shape()

    return pyx

'''
class DataSet(object):
    def __init__(self, images, labels, fake_data=False):

        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSets(object):
        pass

# read in CMU datasets (self-contained option using pickled data)
data = DataSets()
input_file = open('objs.pickle', 'rb')
[imgMat, labMat] = pickle.load(input_file)
input_file.close()

img_mat = imgMat[:1200]
lab_mat = labMat[:1200]
img_mat_T = imgMat[1200:]
lab_mat_T = labMat[1200:]

data.train = DataSet(img_mat, lab_mat)
data.test = DataSet(img_mat_T, lab_mat_T)
trX, trY, teX, teY = data.train.images, data.train.labels, data.test.images, data.test.labels

'''
# read in CMU datasets (import_images option using locally stored images)
mnist = import_images.read_data_sets(one_hot=True) # (removed to create a self-contained file)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, hPix, wPix, 1)
teX = teX.reshape(-1, hPix, wPix, 1)

# tensor placeholders to be used during training and testing
X = tf.placeholder("float", [None, hPix, wPix, 1])
Y = tf.placeholder("float", [None,num_classes])

# shape of filters to be used in convolution layers in model
'''
# stock weights (leads to resources exhausted error)
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, 3])  # HCN
'''
# new weights (trying fewer nodes to conserve memory)
w = init_weights([3, 3, 1, 6])
w2 = init_weights([3, 3, 6, 12])
w3 = init_weights([3, 3, 12, 24])
# l3Shape[1] * l3Shape[2] * l3Shape[3]], 48
w4 = init_weights([33840, 48])
w_o = init_weights([48, 3])

bc1 = tf.Variable(tf.random_normal([6]))
bc2 = tf.Variable(tf.random_normal([12]))
bc3 = tf.Variable(tf.random_normal([24]))
bf1 = tf.Variable(tf.random_normal([48]))
bf2 = tf.Variable(tf.random_normal([3]))

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# somehow, dimensions of py_x doesn't match Y during execution and error is thrown...
# dimension of Y is correct, as it has necessary shape [batch_size, num_classes]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(learning_rate, momentum).minimize(cost)
predict_op = tf.argmax(py_x, 0) # reduces across the 1st dimension of py_x

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(training_iterations):
    # original batch size: 128 (now 64 to conserve memory), original step size 128 (now 32)
    for start, end in zip(range(0, len(trX), batch_increments), range(batch_size, len(trX), batch_increments)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_conv: 0.8, p_keep_hidden: 0.5})

    test_indices = np.arange(len(teX)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:256]

    print i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                     Y: teY[test_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0}))