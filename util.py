import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete


def zeros_like_list(values):
    return [np.zeros(value.get_shape()) for value in values]


def get_dtype(env):
    space = env.observation_space
    if isinstance(space, Discrete):
        return type(space.sample())
    elif isinstance(space, Box):
        return env.observation_space.low.dtype
    else:
        raise ValueError


def get_base_name(var):
    return var.name.split(':')[0]
