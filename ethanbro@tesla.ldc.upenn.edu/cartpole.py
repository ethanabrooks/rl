from functools import partial

import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

import model_zoo
import q_learning
import reinforce

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[1]

env = gym.make('CartPole-v1')

model = partial(model_zoo.mlp, hidden_sizes=[5])
reinforce.train(env, model, optimizer, show_off_at=200)
# q_learning.train(env, model, optimizer)