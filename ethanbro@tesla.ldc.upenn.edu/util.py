import numpy as np


def zeros_like_list(values):
    return [np.zeros(value.get_shape()) for value in values]


def get_dtype(env):
    return env.observation_space.low.dtype


def get_base_name(var):
    return var.name.split(':')[0]
