# coding=utf-8
import random
import shutil

import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
import os


def gather_1d(params, indices):
    h, w = params.get_shape()
    indices = tf.stack([tf.range(h), indices], axis=1)
    return tf.gather_nd(params, indices)


# noinspection PyPep8Naming
def train(env, network,
          optimizer=AdamOptimizer(),
          memory_buffer_size=10000,
          bsize=32,
          epochs=500000,
          gamma=.9,
          print_freq=10,
          epsilon_decay=1000000,
          skip_frames=3,
          summary_dir='summaries'):
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

    assert (isinstance(env.action_space, gym.spaces.Discrete))
    obs_size = env.observation_space.shape
    act_size = env.action_space.n
    dtype = tf.float32
    epsilon = 1

    if type(obs_size) == tuple:
        obs_size = list(obs_size)

    observation_ph = tf.placeholder(dtype,
                                    [1] + obs_size,  # expand to batch size
                                    name='observation')

    obs_rank = len(obs_size)
    if obs_rank in [1, 3]:
        network_input = observation_ph
    elif obs_rank == 2:
        network_input = tf.expand_dims(observation_ph, 3)
    else:
        raise ValueError

    tf_Q = tf.squeeze(network(network_input, act_size, scope='Qnetwork'))
    proposed_action = tf.argmax(tf_Q, axis=0, name='proposed_action')

    observations_ph = []
    actions_ph = tf.placeholder(tf.int32, [bsize], name='batch_actions')
    rewards_ph = tf.placeholder(dtype, [bsize], name='rewards')
    done_ph = tf.placeholder(tf.bool, [bsize], name='done')
    Qs = []
    for _ in range(2):
        batch_observations = tf.placeholder(dtype, [bsize] + obs_size,
                                            name='batch_observations')
        observations_ph.append(batch_observations)
        if obs_rank == 2:
            batch_observations = tf.expand_dims(batch_observations, 3)
        Qs.append(network(batch_observations, act_size, scope='Qnetwork', reuse=True))

    y = tf.where(
        done_ph,
        rewards_ph,
        rewards_ph + gamma * tf.reduce_max(Qs[1], axis=1)
    )
    r = tf.where(
        done_ph,
        tf.zeros_like(rewards_ph),
        rewards_ph
    )

    r_guess = gather_1d(
        network(observations_ph[0], act_size, scope='Rnetwork'), #, reuse=True),
        actions_ph
    )
    y_guess = gather_1d(Qs[0], actions_ph)
    tf_loss = sum(map(tf.square, [y - y_guess, r - r_guess]))
    train_op = optimizer.minimize(tf_loss)

    for var in tf.trainable_variables():
        print(var.name)

    summary_phs = {}
    summary_names = ['reward', 'loss', 'Q']
    for name in summary_names:
        summary_phs[name] = tf.placeholder(tf.float32, (), name)
        tf.summary.scalar(name, summary_phs[name])

    summaries = tf.summary.merge_all()

    show_off = False
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        tf.global_variables_initializer().run()
        memory_buffer = []

        step = 0
        # update every epoch
        for e in range(epochs):
            total_reward = 0
            total_Q = 0
            total_loss = 0
            observation = env.reset()
            done = False
            last_observation = None

            # steps
            memory_buffer_full = False
            while not done:
                step += 1
                if show_off:
                    env.render()
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action, Q = sess.run([proposed_action, tf_Q],
                                         {observation_ph: np.expand_dims(observation, 0)})
                    total_Q += Q[action]

                if epsilon > .1:
                    epsilon *= .1 ** (1. / epsilon_decay)
                for _ in range(skip_frames + 1):
                    next_observation, reward, done, info = env.step(action)

                total_reward += reward

                if last_observation is not None:
                    memory_buffer.append((last_observation, action, reward, next_observation, done))
                last_observation = observation
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
                    total_loss += loss.mean()

            if memory_buffer_full and e % print_freq == 0:
                values = {
                    'reward': total_reward,
                    'loss': total_loss,
                    'Q': total_Q
                }
                writer.add_summary(sess.run(summaries,
                                            {summary_phs[name]: values[name]
                                             for name in summary_names}),
                                   global_step=e)

                print("Epoch: {:5} ϵ: {:.4f}".format(e, epsilon) +
                      ''.join(' {}: {:8.4f}'.format(name, values[name])
                              for name in summary_names))
