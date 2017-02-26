import gym


def get_dtype(env):
    return env.observation_space.low.dtype


def get_base_name(var):
    return var.name.split(':')[0]
