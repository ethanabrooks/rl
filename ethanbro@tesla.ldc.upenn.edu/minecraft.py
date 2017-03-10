from functools import partial

import gym.spaces
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import reinforce
from model_zoo import conv_network
import gym_minecraft

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[1]

env = gym.make('MinecraftBasic-v0')
env.init(start_minecraft=True)

model = partial(conv_network,
                filters=(32, 64),
                kernel_size=(4, 4),
                strides=(2, 2))
reinforce.train(env, model, optimizer)
