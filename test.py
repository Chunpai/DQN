import gym
from time import sleep
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from model import DQN

env = gym.envs.make("Breakout-v0")
ob = env.reset()


tf.reset_default_graph()
# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = DQN(scope="q", summaries_dir=experiment_dir)
target_estimator = DQN(scope="target_q")

