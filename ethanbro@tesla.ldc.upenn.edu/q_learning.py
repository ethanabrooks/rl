# coding=utf-8
import random

import gym.spaces
import tensorflow as tf
import numpy as np

from util import get_dtype


def gather_1d(params, indices):
    h, w = params.get_shape()
    act_indices = tf.stack([tf.range(h), indices], axis=1)
    return tf.gather_nd(params, act_indices)


# noinspection PyPep8Naming
def train(env, network, optimizer,
          bsize=32,
          epochs=50000,
          gamma=.9,
          print_freq=10):
    assert (isinstance(env.action_space, gym.spaces.Discrete))
    obs_size = env.observation_space.shape
    act_size = env.action_space.n
    dtype = tf.float32
    memory_buffer_size = 1000
    epsilon = 1

    if type(obs_size) == tuple:
        obs_size = list(obs_size)
    elif type(obs_size) != list:
        obs_size = [obs_size]

    observation_ph = tf.placeholder(dtype, [1] + obs_size, name='observation')
    tf_Q = network(observation_ph, act_size)
    nominal_action = tf.argmax(tf.squeeze(tf_Q), axis=0, name='nominal_action')

    observations_ph = []
    actions_ph = tf.placeholder(tf.int32, [bsize], name='batch_actions')
    rewards_ph = tf.placeholder(dtype, [bsize], name='reward')
    done_ph = tf.placeholder(tf.bool, [bsize], name='done')
    Qs = []
    for _ in range(2):
        batch_observations = tf.placeholder(dtype, [bsize] + obs_size,
                                            name='batch_observations')
        observations_ph.append(batch_observations)
        Qs.append(network(batch_observations, act_size, reuse=True))

    y = tf.where(
        done_ph,
        rewards_ph,
        rewards_ph + gamma * tf.reduce_max(Qs[1], axis=1)
    )
    y_guess = gather_1d(Qs[0], actions_ph)
    tf_loss = tf.square(y - y_guess)
    train_op = optimizer.minimize(tf_loss)

    for var in tf.trainable_variables():
        print(var)

    show_off = False
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        memory_buffer = []

        step = 0
        # update every epoch
        for e in range(epochs):
            cumulative_reward = 0
            cumulative_Q = 0
            cumulative_loss = 0
            observation = env.reset()
            done = False

            # steps
            memory_buffer_full = False
            while not done:
                step += 1
                if show_off:
                    env.render()
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action, Q = sess.run([nominal_action, tf_Q],
                                         {observation_ph: np.expand_dims(observation, 0)})
                    cumulative_Q += Q[0, action]

                if epsilon > .1:
                    epsilon *= .1 ** (1. / 100000.)
                    # show_off = True
                next_observation, reward, done, info = env.step(action)
                cumulative_reward += reward

                memory_buffer.append((observation, action, reward, next_observation, done))
                observation = next_observation

                memory_buffer_full = len(memory_buffer) >= memory_buffer_size
                if memory_buffer_full:
                    memory_buffer.pop(0)
                    batch = zip(*[random.choice(memory_buffer) for _ in range(bsize)])
                    observations, actions, rewards, next_observations, done_values = batch
                    _, loss = sess.run([train_op, tf_loss],
                                       feed_dict={
                                           observations_ph[0]: observations,
                                           actions_ph: actions,
                                           rewards_ph: rewards,
                                           observations_ph[1]: next_observations,
                                           done_ph: done_values,
                                       })
                    cumulative_loss += loss.mean()

            if memory_buffer_full and e % print_freq == 0:

                print("Epoch: {}. ϵ: {}. Reward: {}. Q: {}. loss: {}"
                      .format(e, epsilon, *(r/epochs for r in
                                            [cumulative_reward,
                                             cumulative_Q,
                                             cumulative_loss])))
