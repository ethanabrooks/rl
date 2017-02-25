import tensorflow as tf


def mlp(x, activation_sizes):
    assert len(activation_sizes) > 0
    for size in activation_sizes[:-1]:
        x = tf.layers.dense(x, size, tf.sigmoid)
    return tf.layers.dense(x, activation_sizes[-1])

def