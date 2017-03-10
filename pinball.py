from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

import q_learning
import reinforce
from model_zoo import trpo_conv_net, dqn_conv_net
from wrapper import EnvWrapper

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[1]

env = EnvWrapper(gym.make('VideoPinball-v0'), .3)
# reinforce.train(env, trpo_conv_net, optimizer, show_off_at=600)
q_learning.train(env, trpo_conv_net, optimizer, print_freq=10)
