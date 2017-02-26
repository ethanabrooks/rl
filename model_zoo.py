from operator import mul

import tensorflow as tf


def mlp(x, out_size, hidden_sizes=None):
    if hidden_sizes is None:
        hidden_sizes = []
    for size in hidden_sizes:
        x = tf.layers.dense(x, size, activation=tf.sigmoid)
    return tf.layers.dense(x, out_size)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_layer(i, x, filter_size, num_filters, stride=1):
    _, _, _, in_channels = x.get_shape()
    filter = tf.get_variable('filter' + str(i),
                             [filter_size, filter_size, in_channels, num_filters],
                             x.dtype)
    return tf.nn.conv2d(x, filter, strides=[1, stride, stride, 1], padding='SAME')


def conv_net(x, out_size, strides, filters_per_layer, filter_size_list, dense_size):
    for i, (filter_size,
            num_filters,
            stride) in enumerate(zip(filter_size_list,
                                     filters_per_layer,
                                     strides)):
        conv_output = conv_layer(i, x, filter_size, num_filters, stride)
        bias = tf.get_variable('bias' + str(i), num_filters)
        x = tf.nn.relu(tf.nn.bias_add(conv_output, bias))

    return mlp(tf.reshape(x, [1, -1]), out_size, hidden_sizes=[dense_size])


def dqn_conv_net(x, out_size):
    return conv_net(x, out_size,
                    strides=(4, 2),
                    filters_per_layer=(16, 32),
                    filter_size_list=(8, 4),
                    dense_size=256)


def trpo_conv_net(x, out_size):
    return conv_net(x, out_size,
                    strides=(2, 2),
                    filters_per_layer=(16, 16),
                    filter_size_list=(4, 4),
                    dense_size=20)

class RecurrentConv:
    def __init__(self, cell=tf.contrib.rnn.GRUBlockCell, ):
        self._cell = cell

    def forward(self, x, out_size):

