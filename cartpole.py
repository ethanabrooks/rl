from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

import model_zoo
import q_learning

env = gym.make('CartPole-v1')

model = partial(model_zoo.mlp, hidden_sizes=[5])
# reinforce.train(env, model, optimizer, show_off_at=200)
q_learning.train(env, model, print_freq=200, epsilon_decay=1000000, skip_frames=0)