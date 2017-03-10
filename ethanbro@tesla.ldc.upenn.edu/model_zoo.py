from operator import mul

import tensorflow as tf


def mlp(x, out_size, hidden_sizes=None, reuse=False):
    if hidden_sizes is None:
        hidden_sizes = []
    for i, size in enumerate(hidden_sizes):
        x = tf.layers.dense(x, size,
                            activation=tf.sigmoid,
                            name="layer" + str(i),
                            reuse=reuse)
    return tf.layers.dense(x, out_size,
                           name="out_layer",
                           reuse=reuse)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_layer(i, x, filter_size, num_filters, stride=1):
    if len(x.get_shape()) < 4:
        x = tf.expand_dims(x, 3)
    in_channels = x.get_shape()[-1]
    filter = tf.get_variable('filter' + str(i),
                             [filter_size, filter_size, in_channels, num_filters])
    return tf.nn.conv2d(x, filter, strides=[1, stride, stride, 1], padding='SAME')


def conv_net(x, out_size, strides, filters_per_layer, filter_size_list, dense_size, reuse=False):
    for i, (filter_size,
            num_filters,
            stride) in enumerate(zip(filter_size_list,
                                     filters_per_layer,
                                     strides)):
        with tf.variable_scope('conv_net', reuse=reuse):
            conv_output = conv_layer(i, x, filter_size, num_filters, stride)
            bias = tf.get_variable('bias' + str(i), num_filters)
            add = tf.nn.bias_add(conv_output, bias)
            x = tf.nn.relu(add)

    batch_size = int(x.get_shape()[0])
    return mlp(tf.reshape(x, [batch_size, -1]), out_size, hidden_sizes=[dense_size], reuse=reuse)


def dqn_conv_net(x, out_size):
    return conv_net(x, out_size,
                    strides=(4, 2),
                    filters_per_layer=(16, 32),
                    filter_size_list=(8, 4),
                    dense_size=256)


def trpo_conv_net(x, out_size, reuse=False):
    return conv_net(x, out_size,
                    strides=(2, 2),
                    filters_per_layer=(16, 16),
                    filter_size_list=(4, 4),
                    dense_size=20,
                    reuse=reuse)


class RecurrentConv:
    def __init__(self,
                 cell=tf.contrib.rnn.GRUBlockCell(10),
                 conv_net=trpo_conv_net):
        self._cell = cell
        self._state = None
        self._conv_net = conv_net

    def forward(self, x, out_size):
        if self._state is None:
            batch_size = x.get_shape()[0]
            self._state = tf.get_variable('state', [batch_size, self._cell.state_size])
            tf.global_variables_initializer()

        conv_out = self._conv_net(x, self._cell.state_size)  # arbitrary size
        cell_out, self._state = self._cell(conv_out, self._state)
        return mlp(cell_out, out_size)


class MultiStepConv:
    def __init__(self,
                 cell=tf.contrib.rnn.GRUBlockCell(10),
                 conv_net=trpo_conv_net,
                 bptt_steps=6):
        self._cell = cell
        self._states = []
        self._bptt_steps = bptt_steps
        self._conv_net = conv_net
        self._cell_state = None

    def forward(self, x, out_size):

        # initialize cell state
        if self._cell_state is None:
            batch_size = x.get_shape()[0]
            self._cell_state = tf.get_variable('cell_state', [batch_size, self._cell.state_size])
            tf.global_variables_initializer()

        self._states.append(self._conv_net(x, self._cell.state_size))
        if len(self._states) > self._bptt_steps:
            self._states.pop(0)

        # progress last cell state for backpropogation by one
        cell_out, cell_state = self._cell(self._states[0], self._cell_state)
        self._cell_state = tf.stop_gradient(cell_state)
        for conv_state in self._states[1:]:
            cell_out, cell_state = self._cell(conv_state, cell_state)

        return mlp(cell_out, out_size)


class StatefulGRU:
    def __init__(self, size):
        self._state = None
        self._cell = tf.contrib.rnn.GRUBlockCell(size)

    def forward(self, x, out_size):
        if self._state is None:
            batch_size = x.get_shape()[0]
            self._state = tf.get_variable('cell_state', [batch_size, self._cell.state_size])
        x, self._state = self._cell(x, self._state)
        return mlp(x, out_size)
