from __future__ import print_function

import gym.spaces
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import *


def train(env, network, optimizer, show_off_at):
    dtype = tf.float32
    assert (isinstance(env.action_space, gym.spaces.Discrete))
    obs_size = env.observation_space.shape
    act_size = env.action_space.n

    # get action
    observation_ph = tf.placeholder(dtype, obs_size, name='observation')
    logits = network(x=tf.expand_dims(observation_ph, 0), out_size=act_size)
    action_dist = tf.nn.softmax(logits, name='action_dist')
    tf_action = tf.squeeze(tf.multinomial(logits, 1), name='action')

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for param in params:
        print(param)

    # get score
    prob = tf.gather(tf.squeeze(action_dist), tf_action, name='prob')
    tf_scores = tf.gradients(tf.log(prob), params, name='gradient')

    # apply gradient
    gradient_phs = [tf.placeholder(dtype, param.get_shape(),
                                   name=get_base_name(param) + '_placeholder')
                    for param in params]
    train_op = optimizer.apply_gradients(zip(gradient_phs, params))

    epochs = 200
    batches = 500

    show_off = False
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # update every epoch
        for e in range(epochs):
            gradients = zeros_like_list(params)
            mean_reward = 0

            # average over batches
            for b in tqdm(range(batches)):
                observation = env.reset()
                done = False
                t = 0
                cumulative_reward = 0
                cumulative_scores = zeros_like_list(params)

                # steps
                while not done:
                    if show_off:
                        env.render()
                    action, new_scores = sess.run([tf_action, tf_scores],
                                                  {observation_ph: observation.squeeze()})
                    observation, reward, done, info = env.step(action)

                    cumulative_reward += reward
                    for old_score, new_score in zip(cumulative_scores, new_scores):
                        old_score += new_score
                    t += 1

                mean_reward += cumulative_reward / batches
                for gradient, cumulative_score in zip(gradients, cumulative_scores):
                    gradient -= cumulative_score * cumulative_reward / batches

            print("\rEpoch: {}. Reward: {}".format(e, mean_reward))
            feed_dict = dict(zip(gradient_phs, gradients))
            sess.run(train_op, feed_dict)
            if mean_reward >= show_off_at:
                show_off = True
