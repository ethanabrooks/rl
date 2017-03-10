from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import reinforce
from model_zoo import trpo_conv_net, dqn_conv_net, RecurrentConv, MultiStepConv

env = gym.make('Pong-v0')
model = RecurrentConv()
reinforce.train(env, model.forward, show_off_at=600)
