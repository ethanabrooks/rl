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
def train(env, network, optimizer):
    assert (isinstance(env.action_space, gym.spaces.Discrete))
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    dtype = get_dtype(env)

    bsize = 32
    memory_buffer_size = 3 * bsize
    epochs = 50000
    gamma = .9
    epsilon = .5

    observation_ph = tf.placeholder(dtype, [1, obs_size], name='observation')
    tf_Q = network(observation_ph, act_size)
    nominal_action = tf.argmax(tf.squeeze(tf_Q), axis=0, name='nominal_action')

    observations_ph = []
    actions_ph = tf.placeholder(tf.int32, [bsize], name='batch_actions')
    rewards_ph = tf.placeholder(dtype, [bsize], name='reward')
    Qs = []
    for _ in range(2):
        batch_observations = tf.placeholder(dtype, [bsize, obs_size],
                                            name='batch_observations')
        observations_ph.append(batch_observations)
        Qs.append(network(batch_observations, act_size))

    y = rewards_ph + gamma * tf.reduce_max(Qs[1], axis=1)
    d = gather_1d(Qs[0], actions_ph)
    tf_loss = tf.square(y - d)
    train_op = optimizer.minimize(tf_loss)

    show_off = False
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        memory_buffer = []

        # update every epoch
        for e in range(epochs):
            mean_reward = 0
            observation = env.reset()
            done = False

            step = 0
            # steps
            while not done:
                step += 1
                if show_off:
                    env.render()
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action, Q = sess.run([nominal_action, tf_Q],
                                         {observation_ph: observation.reshape(1, -1)})

                next_observation, reward, done, info = env.step(action)
                if done:
                    next_observation = None
                mean_reward = mean_reward * (step - 1) / step + reward / step
                memory_buffer.append((observation, action, reward, next_observation))
                observation = next_observation

                if len(memory_buffer) >= memory_buffer_size:
                    random.shuffle(memory_buffer)
                    batch, memory_buffer = memory_buffer[:bsize], memory_buffer[bsize:]
                    sequence = zip(*batch)
                    observations, actions, rewards, next_observations = map(np.array, sequence)
                    _, loss = sess.run([train_op, tf_loss], feed_dict={
                        observations_ph[0]: observations,
                        actions_ph: actions,
                        rewards_ph: rewards,
                        observations_ph[1]: next_observations,
                    })

                    # print("Epoch: {}. Reward: {}. loss: {}".format(e, mean_reward, loss.mean()))
