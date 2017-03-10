from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

import q_learning
import reinforce
from model_zoo import trpo_conv_net, dqn_conv_net
from wrapper import EnvWrapper

env = EnvWrapper(gym.make('VideoPinball-v0'), scale=.3,
                 buffer_size=4,
                 crop_top=45,
                 crop_bottom=30,
                 crop_left=10,
                 crop_right=10)
# reinforce.train(env, trpo_conv_net, optimizer, show_off_at=600)
q_learning.train(env, dqn_conv_net, print_freq=1)
