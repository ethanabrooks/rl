# coding=utf-8
import random

import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer


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
          summary_dir='qr-learning'):

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

    r_scope = 'R_network'
    q_scope = 'Q_network'
    tf_Q, tf_r = (tf.squeeze(network(network_input, act_size, scope=scope))
                  for scope in [q_scope, r_scope])
    proposed_action = tf.argmax(tf_r + gamma * tf_Q, axis=0, name='proposed_action')

    observations_ph = []
    rewards_ph = []
    done_ph = []
    Qs = []
    actions_ph = tf.placeholder(tf.int32, [bsize], name='batch_actions')

    for _ in range(2):
        batch_observations = tf.placeholder(dtype, [bsize] + obs_size,
                                            name='batch_observations')
        observations_ph.append(batch_observations)
        if obs_rank == 2:
            batch_observations = tf.expand_dims(batch_observations, 3)
        Qs.append(network(batch_observations, act_size, scope=q_scope, reuse=True))
        rewards_ph.append(tf.placeholder(dtype, [bsize], name='rewards'))
        done_ph.append(tf.placeholder(tf.bool, [bsize], name='done'))

    not_dones = [1 - tf.to_float(done) for done in done_ph]
    y = not_dones[0] * (  # if not done on step 0
        rewards_ph[1] + not_dones[1] * (  # if not done on step 1
            gamma * tf.reduce_max(Qs[1], axis=1)
        )
    )
    r = not_dones[0] * rewards_ph[0]

    r_guess = gather_1d(
        network(observations_ph[0], act_size, scope=r_scope, reuse=True),
        actions_ph
    )
    y_guess = gather_1d(Qs[0], actions_ph)
    y_loss, r_loss = map(tf.square, [y - y_guess, r - r_guess])
    tf_loss = y_loss + r_loss
    train_op = optimizer.minimize(tf_loss)

    for var in tf.trainable_variables():
        print(var.name)

    names = ['reward', 'loss', 'Q', 'r_loss']
    summary_phs = {}
    for name in names:
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
            total_r_loss = 0
            observation = env.reset()
            done = False
            last_values = None

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

                current_values = [observation, action, reward, done]
                if last_values is not None:
                    memory_buffer.append(last_values + current_values)
                if done:
                    memory_buffer.append(current_values + [next_observation, 0, 0, True])
                last_values = None if done else current_values
                observation = next_observation

                memory_buffer_full = len(memory_buffer) >= memory_buffer_size
                if memory_buffer_full:
                    memory_buffer.pop(0)
                    batch = zip(*[random.choice(memory_buffer) for _ in range(bsize)])
                    _, loss, r_loss_, r_, r_guess_ = sess.run(
                        [train_op, tf_loss, r_loss, r, r_guess],
                        feed_dict={
                            observations_ph[0]: batch[0],
                            observations_ph[1]: batch[4],
                            actions_ph: batch[1],
                            rewards_ph[0]: batch[2],
                            rewards_ph[1]: batch[6],
                            done_ph[0]: batch[3],
                            done_ph[1]: batch[7],
                        })
                    total_loss += loss.mean()
                    total_r_loss += r_loss_.mean()

                    # dones = batch[3]
                    # if True in dones:
                    #     print(batch[2])
                    #     print(batch[3])
                    #     print(r_)
                    #     print(batch[3].index(True))

            if memory_buffer_full and e % print_freq == 0:
                # names = ['reward', 'loss', 'Q', 'r loss']
                values = [eval('total_' + name) for name in names]
                pairs = zip(names, values)
                writer.add_summary(global_step=e,
                                   summary=sess.run(summaries,
                                                    {summary_phs[name]: value
                                                     for name, value in pairs}))
                print("Epoch: {:5} ϵ: {:.4f}".format(e, epsilon, total_r_loss) +
                      ''.join(' {}: {:8.4f}'.format(name, value)
                              for name, value in pairs))
                # if not all(r_):
                #     tuples = zip(r_, r_guess_)
                #     print('terminal guess: {}'
                #           .format(filter(lambda (r, _): not r, tuples)))
                #     print('avg for non-terminal: {}'
                #           .format(np.mean([g for __r, g in tuples if bool(__r)])))

