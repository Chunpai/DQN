from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import gym
from gym.wrappers import Monitor
import plotting
from collections import deque, namedtuple


print(tf.__version__)
env = gym.envs.make("Breakout-v0")

# ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
# ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
# print('Elements of ds_tensors:')
# for x in ds_tensors:
#     print(x)

def state_process(input_state):
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    :param input_state: A [210, 160, 3] Atari RGB State
    :return: A processed [84, 84] state representing grayscale values.
    """
    output = tf.image.rgb_to_grayscale(input_state)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output = tf.squeeze(output)
    return output


VALID_ACTIONS = [0, 1, 2, 3]

model = keras.Sequential([keras.layers.Conv2D(32, (8, 8), 4,
                                              input_shape=(84, 84, 4),
                                              activation='relu',
                                              padding='same'),
                          keras.layers.Conv2D(64, (4, 4), 2,
                                              activation='relu',
                                              padding='same'),
                          keras.layers.Conv2D(64, (3, 3), 1,
                                              activation='relu',
                                              padding='same'),
                          keras.layers.Flatten(),
                          keras.layers.Dense(512),
                          keras.layers.Dense(len(VALID_ACTIONS))

                          ])
model.summary()  # we need to specify the input shape in order to see the summary.


# output = model(train_input)

def loss(q_output, td_target):
    """
    loss of regression
    :param q_output: output of network Q
    :param td_target: output of target network Q hat
    :return: mean square of (q_output - td_target)
    """
    return tf.reduce_mean(tf.square(q_output - td_target))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def deep_q_learning():
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # The replay memory
    replay_memory = []
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    return
