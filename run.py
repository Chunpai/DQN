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
<<<<<<< HEAD
from estimator import Estimator

print(tf.__version__)
=======
from model import Estimator

print(tf.__version__)
env = gym.envs.make("Breakout-v0")

>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6

# ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
# ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
# print('Elements of ds_tensors:')
# for x in ds_tensors:
#     print(x)

<<<<<<< HEAD
VALID_ACTIONS = [0, 1, 2, 3]


@tf.function
=======
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
def state_process(input_state):
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    :param input_state: A [210, 160, 3] Atari RGB State
    :return: A processed [84, 84] state representing grayscale values.
    """
<<<<<<< HEAD
    # print("shape of input_state {}".format(input_state.shape))
=======
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
    output = tf.image.rgb_to_grayscale(input_state)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output = tf.squeeze(output)
    return output


<<<<<<< HEAD
@tf.function
def loss_grad(model, state_batch, action_batch, target_batch):
=======


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
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
    """
    compute the loss gradient w.r.t to all trainable variables in the model
    :param model: we will update the q_estimator to approximate the TD Target
    :param inputs: input of the model, state
    :param targets: the TD target
    :return: loss value and the loss gradient
    """
<<<<<<< HEAD

    state_batch = tf.cast(state_batch, tf.float32)
    action_batch = tf.cast(action_batch, tf.int32)
    target_batch = tf.cast(target_batch, tf.float32)
    with tf.GradientTape() as tape:  # note: everything should be tensor in the tape
        state_batch = state_batch / 255.0  # we should cast to tf.float32 then do the division
        v_values = model(tf.cast(state_batch / 255.0, tf.float32))
        gather_indices = tf.range(tf.shape(v_values)[0]) * tf.shape(v_values)[1] + action_batch
        # flat the v_values to 1D array, then gather
        q_values_batch = tf.gather(tf.reshape(v_values, [-1]), gather_indices)
        losses = tf.keras.metrics.mean_squared_error(y_true=target_batch, y_pred=q_values_batch)
    gradients = tape.gradient(losses, model.trainable_variables)
    loss = tf.reduce_mean(losses)
    # optimizer = tf.keras.optimizers.RMSprop(lr=0.00025, decay=0.99, rho=0, epsilon=1e-6)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gradients
=======
    with tf.GradientTape() as tape:
        loss_value = loss(inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6


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
<<<<<<< HEAD
    q_values = estimator(np.expand_dims(state, 0).astype(np.float32) / 255.0)[0]
=======
    q_values = estimator(np.expand_dims(state, 0))[0]
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A


<<<<<<< HEAD
@tf.function   # we must use tf.print to print in this decoration function
def copy_model_parameters(estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    print("all trainable variables est1 {}".format([v.name for v in estimator1.trainable_variables]))
    print("all trainable variables est2 {}".format([v.name for v in estimator2.trainable_variables]))
    e1_params = [t for t in estimator1.trainable_variables]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in estimator2.trainable_variables]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    for e1_v, e2_v in zip(e1_params, e2_params):
        # tf.print("e1_v name: {}".format(e1_v.name))
        # tf.print(e1_v)
        # tf.print("e2_v name : {}".format(e2_v.name))
        # tf.print(e2_v, output_stream=sys.stdout)
        e2_v.assign(e1_v)
        # tf.print("e2_v name after: {}".format(e2_v.name))
        # tf.print(e2_v, output_stream=sys.stdout)


def deep_q_learning(env, q_estimator, target_estimator, log_dir,
                    num_episodes=10000,
                    replay_buffer_size=500000,
                    replay_buffer_init_size=50000,
                    update_target_estimator_every=10000,
=======
def deep_q_learning(env, q_estimator, target_estimator, num_episodes, log_dir,
                    replay_buffer_size=500000,
                    replay_buffer_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
<<<<<<< HEAD
                    discount_factor=0.99,
                    record_video_every=50):

    tf.random.set_seed(123)
    np.random.seed(123)

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_buffer = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    summary_writer = tf.summary.create_file_writer(log_dir)
    # # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    # monitor_path = os.path.join(log_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # if not os.path.exists(monitor_path):
    #     os.makedirs(monitor_path)
=======
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
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6

    # saver = tf.train.Saver()
    # # Load a previous checkpoint if we find one
    # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # if latest_checkpoint:
    #     print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    #     saver.restore(latest_checkpoint)

<<<<<<< HEAD
    total_t = 0
    num_actions = len(VALID_ACTIONS)
=======
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
    # epsilon decay scheduling
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    state = env.reset()
    state = state_process(state)
    state = np.stack([state] * 4, axis=2)  # shape (84, 84, 4)
<<<<<<< HEAD
    # initialize the replay buffer
    for i in range(replay_buffer_init_size):
        if i % 2 == 0:
            action_probs = behavior_policy(q_estimator, state, epsilons[total_t], num_actions)
        else: # the reason to do this is to initialize the variable in target_estimator, thus we can copy parameters
            action_probs = behavior_policy(target_estimator, state, epsilons[total_t], num_actions)

=======

    total_t = 0
    num_actions = len(VALID_ACTIONS)
    # initialize the replay buffer
    for i in range(replay_buffer_init_size):
        action_probs = behavior_policy(q_estimator, state, epsilons[total_t], num_actions)
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
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
<<<<<<< HEAD
    print("Done initialization of the replay buffer !")
    # Record videos using the gum env monitor wrapper
    # env = Monitor()

    # simply follow the pseudocode of DQN algorithm
    for i_episode in range(num_episodes):
        state = env.reset()  # reset the environment for every episode
        state = state_process(state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        # one step in environment, and iteratively generate a full episode.
        for t in itertools.count():  # same as while loop
            # get epsilon for current step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            # initialize the target network and
            # update the target estimator every some steps
            if total_t % update_target_estimator_every == 0:
                # print("q_estimator weight {}".format(q_estimator.get_weights()))
                # print("target estimator weight before {}".format(target_estimator.get_weights()))
                # input_tensors = keras.layers.Input(shape=(84, 84, 3))
                # target_estimator = keras.models.clone_model(q_estimator, input_tensors=input_tensors )
                # target_estimator.set_weights(q_estimator.get_weights())
                copy_model_parameters(q_estimator, target_estimator)

                # print("target estimator weight after {}".format(target_estimator.get_weights()))
                print("\nCopied model parameters to target network.")
            print("\rStep: {} ({}) at Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss), end="")
            # print("Step: {} ({}) at Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss))
=======

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

>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
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
<<<<<<< HEAD
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            samples = random.sample(replay_buffer, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))  # * used to unpack the seq.

            # Calculate target_values using DQN
            # q_values_next = target_estimator(next_states_batch.astype(np.float32) / 255.0)
            # target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)
            # Calculate target_values using double DQN
            q_values_next = q_estimator(next_states_batch.astype(np.float32) / 255.0)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator(next_states_batch.astype(np.float32) / 255.0)
            gather_indices = np.arange(batch_size) * len(VALID_ACTIONS) + best_actions
            max_q_values_next_target = tf.gather(tf.reshape(q_values_next_target, [-1]), gather_indices)
            target_batch = reward_batch + np.invert(done_batch).astype(np.float32) \
                           * discount_factor * max_q_values_next_target


            # loss = train(q_estimator, states_batch, action_batch, target_batch)
            # optimize and reduce the loss
            loss, grads = loss_grad(q_estimator, states_batch, action_batch, target_batch)
            optimizer = tf.keras.optimizers.RMSprop(lr=0.00025, decay=0.99, rho=0, epsilon=1e-6)
            optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))

            # # Summaries for Tensorboard
            # with summary_writer.as_default():
            #     tf.summary.scalar("loss", loss, step=total_t)

=======
            samples = random.sample(replay_buffer, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))  # * used to unpack the seq.

            # Calculate q values and targets (Double DQN)
            target_values = target_estimator(next_states_batch)
            # q_values = q_estimator.predict(states_batch)
            # Perform gradient descent update
            # states_batch = np.array(states_batch)
            loss_value, grads = loss_grad(q_estimator, states_batch, target_values)
            optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))

>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
            if done:
                break
            state = next_state
            total_t += 1
<<<<<<< HEAD

        # # Add summaries to tensorboard
        # with summary_writer.as_default():
        #     tf.summary.scalar("episode/epsilon", epsilon, step=i_episode)
        #     tf.summary.scalar("episode/reward", stats.episode_rewards[i_episode], step=i_episode)
        #     tf.summary.scalar("episode/length", stats.episode_lengths[i_episode], step=i_episode)

        yield total_t, i_episode, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])

    return stats


# q_estimator = keras.Sequential([keras.layers.Input(shape=(84, 84, 4), dtype=tf.float32),
#                                 keras.layers.Conv2D(32, (8, 8), 4,
#                                                     activation='relu',
#                                                     padding='same'),
#                                 keras.layers.Conv2D(64, (4, 4), 2,
#                                                     activation='relu',
#                                                     padding='same'),
#                                 keras.layers.Conv2D(64, (3, 3), 1,
#                                                     activation='relu',
#                                                     padding='same'),
#                                 keras.layers.Flatten(),
#                                 keras.layers.Dense(512),
#                                 keras.layers.Dense(len(VALID_ACTIONS))
#                                 ])

# @tf.function
# def Estimator():
#     model = keras.Sequential()
#     model.add(keras.layers.Input(shape=(84, 84, 4), dtype=tf.float32))
#     model.add(keras.layers.Conv2D(32, (8, 8), 4, activation='relu', padding='same'))
#     model.add(keras.layers.Conv2D(64, (4, 4), 2, activation='relu', padding='same'))
#     model.add(keras.layers.Conv2D(64, (3, 3), 2, activation='relu', padding='same'))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(512))
#     model.add(keras.layers.Dense(len(VALID_ACTIONS)))
#     return model



# target_estimator = keras.Sequential([keras.layers.Conv2D(32, (8, 8), 4,
#                                                          input_shape=(84, 84, 4),
#                                                          activation='relu',
#                                                          padding='same'),
#                                      keras.layers.Conv2D(64, (4, 4), 2,
#                                                          activation='relu',
#                                                          padding='same'),
#                                      keras.layers.Conv2D(64, (3, 3), 1,
#                                                          activation='relu',
#                                                          padding='same'),
#                                      keras.layers.Flatten(),
#                                      keras.layers.Dense(512),
#                                      keras.layers.Dense(len(VALID_ACTIONS))
#                                      ])

q_estimator = Estimator(len(VALID_ACTIONS))
target_estimator = Estimator(len(VALID_ACTIONS))
print("all trainable variables q est {}".format([v.name for v in q_estimator.trainable_variables]))
print("all trainable variables target est {}".format([v.name for v in target_estimator.trainable_variables]))


# q_estimator.summary()
env = gym.make("Breakout-v0")
# obs = env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#
#     if done:
#         obs = env.reset()
# env.close()


log_dir = os.path.abspath("./logs/{}".format(env.spec.id))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)



for t, ep, stats in deep_q_learning(env, q_estimator, target_estimator, log_dir,
                                num_episodes=10000,
                                replay_buffer_size=500000,
                                # replay_buffer_init_size=50000,
                                replay_buffer_init_size=50,
                                update_target_estimator_every=10000,
                                epsilon_start=1.0,
                                epsilon_end=0.1,
                                epsilon_decay_steps=500000,
                                batch_size=32,
                                discount_factor=0.99):
    print("\nEpisode Reward: {}, {}".format(ep, stats.episode_rewards[-1]))
=======
    return
>>>>>>> 6f4843efe15ffe7397da609412ee478ef71ba9b6
