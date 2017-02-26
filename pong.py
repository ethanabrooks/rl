from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import reinforce
from model_zoo import trpo_conv_net, dqn_conv_net, RecurrentConv, MultiStepConv

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[1]

env = gym.make('Pong-v0')
model = MultiStepConv()
reinforce.train(env, model.forward, optimizer, show_off_at=600)
