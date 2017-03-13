from functools import partial

import gym.spaces

import model_zoo
import q_learning

env = gym.make('CartPole-v1')

# model = partial(model_zoo.mlp, hidden_sizes=[5])
model = model_zoo.simple_conv_net
# reinforce.train(env, model, optimizer, show_off_at=200)
q_learning.train(env, model, print_freq=200, epsilon_decay=1000000, skip_frames=0)