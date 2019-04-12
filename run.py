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
from model import Estimator

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




# model = keras.Sequential([keras.layers.Conv2D(32, (8, 8), 4,
#                                               input_shape=(84, 84, 4),
#                                               activation='relu',
#                                               padding='same'),
#                           keras.layers.Conv2D(64, (4, 4), 2,
#                                               activation='relu',
#                                               padding='same'),
#                           keras.layers.Conv2D(64, (3, 3), 1,
#                                               activation='relu',
#                                               padding='same'),
#                           keras.layers.Flatten(),
#                           keras.layers.Dense(512),
#                           keras.layers.Dense(len(VALID_ACTIONS))
#                           ])
# model.summary()  # we need to specify the input shape in order to see the summary.

VALID_ACTIONS = [0, 1, 2, 3]
q_estimator = Estimator(len(VALID_ACTIONS))
target_estimator = Estimator(len(VALID_ACTIONS))


# output = model(train_input)

def loss(model, x, y):
    """
    loss of regression
    :param x: input of network Q
    :param y: output of target network Q hat
    :return: mean square of (q_output - td_target)
    """
    loss_object = tf.keras.losses.mean_squared_error(from_logits=True)
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


def loss_grad(model, inputs, targets):
    """
    compute the loss gradient w.r.t to all trainable variables in the model
    :param model: we will update the q_estimator to approximate the TD Target
    :param inputs: input of the model, state
    :param targets: the TD target
    :return: loss value and the loss gradient
    """
    with tf.GradientTape() as tape:
        loss_value = loss(inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def behavior_policy(estimator, state, epsilon, num_actions):
    """
    create a behavior policy (epsilon-greedy) based on a given Q-function estimator
    :param estimator: q_estimator
    :param state: the input state
    :param epsilon: the current epsilon
    :param num_actions: size of action space
    :return:
    """
    A = np.ones(num_actions, dtype=float) * epsilon / num_actions
    q_values = estimator(np.expand_dims(state, 0))[0]
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A


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
    state = env.reset()
    state = state_process(state)
    state = np.stack([state] * 4, axis=2)  # shape (84, 84, 4)

    total_t = 0
    num_actions = len(VALID_ACTIONS)
    # initialize the replay buffer
    for i in range(replay_buffer_init_size):
        action_probs = behavior_policy(q_estimator, state, epsilons[total_t], num_actions)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # simply follow the pseudocode of DQN algorithm
    for i_episode in range(num_episodes):
        # reset the environment for every episode
        state = env.reset()
        state = state_process(state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # one step in environment, and iteratively generate full episode as well.
        for t in itertools.count():  # same as while loop
            # get epsilon for current step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            # update the target estimator every some steps
            if current_decay_step % update_target_estimator_every == 0:
                target_estimator = keras.models.clone_model(q_estimator)
                target_estimator.set_weights(q_estimator.get_weights())
                print("\nCopied model parameters to target network.")

            sys.stdout.flush()

            # Take a step
            action_probs = behavior_policy(q_estimator, state, epsilon, num_actions)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            if len(replay_buffer) == replay_buffer_size:
                replay_buffer.pop(0)

            replay_buffer.append(Transition(state, action, reward, next_state, done))
            samples = random.sample(replay_buffer, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))  # * used to unpack the seq.

            # Calculate q values and targets (Double DQN)
            target_values = target_estimator(next_states_batch)
            # q_values = q_estimator.predict(states_batch)
            # Perform gradient descent update
            # states_batch = np.array(states_batch)
            loss_value, grads = loss_grad(q_estimator, states_batch, target_values)
            optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))

            if done:
                break
            state = next_state
            total_t += 1
    return
