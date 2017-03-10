import gym.spaces

import reinforce
from model_zoo import trpo_conv_net

env = gym.make('Qbert-v0')
reinforce.train(env, trpo_conv_net, show_off_at=600)
