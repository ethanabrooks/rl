import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

import model_zoo
from model_zoo import mlp

# env = [10, -1, -1, -1, -1, 1]
env = [10, -1, 1]

LEFT = -1
RIGHT = 1
actions = [LEFT, RIGHT]


def clip(value, low, high):
    return max(min(value, high), low)


def step(action_choice, state):
    nominal_choice = state + actions[int(action_choice)]
    new_state = clip(nominal_choice, 0, len(env) - 1)
    return new_state, env[new_state], new_state == 0


embeddings = tf.get_variable('embeddings', [len(env), 10])
model = model_zoo.StatefulGRU(10)

dtype = tf.int32
tf_state = tf.placeholder(dtype, (), name='state')
lookup = tf.nn.embedding_lookup(embeddings, tf_state)

act_size = len(actions)
tf_values = model.forward(tf.expand_dims(lookup, 0), act_size)
tf_action = tf.squeeze(tf.multinomial(tf_values, 1))

tf_reward = tf.placeholder(tf.float32, (), name='reward')
value = tf.gather(tf.squeeze(tf_values), tf_action)
loss = tf.square(value - tf_reward)

train_op = AdamOptimizer(learning_rate=.1).minimize(loss)


def train(epochs=10):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            state = 1
            done = False
            actions = []
            rewards = []
            states = []

            for _ in range(20):
                action, values = sess.run([tf_action, tf_values], {tf_state: state})
                actions.append(action)
                state, reward, done = step(action, state)
                states.append(state)
                rewards.append(reward)
                gradients = sess.run(train_op, {tf_state: state,
                                    tf_action: action,
                                    tf_reward: reward})
                if done:
                    break
            print(env)
            print(actions)


train()
