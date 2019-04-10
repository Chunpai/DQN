from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import gym
from gym.wrappers import Monitor
import plotting
from collections import deque, namedtuple
import itertools
import random

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


def predict():
    return


def behavior_policy(estimator, num_actions):
    """
    create a behavior policy (epsilon-greedy) based on a given Q-function estimator
    :param estimator:
    :param num_actions:
    :return:
    """
    return


def deep_q_learning(env, q_estimator, target_estimator, num_episodes, log_dir,
                    replay_buffer_size=500000,
                    replay_buffer_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_buffer = []
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(log_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    # saver = tf.train.Saver()
    # # Load a previous checkpoint if we find one
    # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # if latest_checkpoint:
    #     print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    #     saver.restore(latest_checkpoint)

    # epsilon decay scheduling
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = behavior_policy(q_estimator, len(VALID_ACTIONS))
    state = env.reset()
    state = state_process(state)
    state = np.stack([state] * 4, axis=2)  # shape (84, 84, 4)

    # initialize the replay buffer
    for i in range(replay_buffer_init_size):
        action_probs = policy(state, epsilons[0])
        action = np.random.choice(VALID_ACTIONS, p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_process(next_state)
        # append (84, 84, 3) with (84, 84, 1)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_buffer.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_process(state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    # Record videos using the gum env monitor wrapper
    # env = Monitor()

    current_decay_step = 0
    # simply follow the pseudocode of DQN algorithm
    for i_episode in range(num_episodes):
        # reset the environment for every episode
        state = env.reset()
        state = state_process(state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # one step in environment, and iteratively generate full episode as well.
        for t in itertools.count(): # same as while loop
            # get epsilon for current step
            epsilon = epsilons[min(current_decay_step, epsilon_decay_steps-1)]
            # update the target estimator every some steps
            if current_decay_step % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            sys.stdout.flush()

            # Take a step
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            if len(replay_buffer) == replay_buffer_size:
                replay_buffer.pop(0)

            replay_buffer.append(Transition(state, action, reward, next_state, done))

            samples = random.sample(replay_buffer, batch_size)


    return
