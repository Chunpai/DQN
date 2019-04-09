import gym
from time import sleep

env = gym.make('MsPacman-v0')
ob = env.reset()
for _ in range(2000):
    # env.render()
    action = env.action_space.sample()
    ob_next, reward, done, info = env.step(action) # take a random action
    print("time: {}, action: {}, reward: {}".format(_, action, reward))
    if done:
        print("Episode finished after {} timesteps, reward: {}".format(_, reward))
        break
env.close()

# import tensorflow
# import matplotlib
# # matplotlib.use('GTKAgg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.plot(np.arange(100))
# plt.show()
