# import tensorflow as tf
# import gym
# from time import sleep
# from gym.wrappers import Monitor
# import itertools
# import numpy as np
# import os
# import random
# import sys
# from model import DQN
#
# env = gym.envs.make("Breakout-v0")
# ob = env.reset()
#
# # tf.reset_default_graph()
# # Where we save our checkpoints and graphs
# log_dir = os.path.abspath("./logs/{}".format(env.spec.id))
# # Create a glboal step variable
# global_step = tf.train.get_global_step(0, name='global_step', trainable=False)
#
# # Create estimators
# q_estimator = DQN(scope="q", summaries_dir=log_dir)
# target_estimator = DQN(scope="target_q")

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train size {}, label size {}".format(train_images.shape, train_labels.shape))


# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()


# model.compile(optimizer='adam',
#           loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# x_val = train_images[:10000]
# partial_train_images = train_images[10000:]
# y_val = train_labels[:10000]
# partial_train_labels = train_labels[10000:]
#
# history = model.fit(partial_train_images, partial_train_labels, epochs=5, batch_size=50,
#           validation_data=(x_val, y_val), verbose=1)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('\nTest accuracy:', test_acc)
#
# predictions = model.predict(test_images)
# print("pred label {}".format(np.argmax(predictions[0])))
#
# history_dict = history.history
# print(history_dict.keys())

"""
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.square(2) + tf.square(3))

import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))
print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


x = tf.random.uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

import time


def time_matmul(x):
    start = time.time()
    for loop in range(1000):
        tf.matmul(x, x)
    result = time.time() - start
    print("10 loops: {:0.2f}ms".format(1000 * result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
"""

"""
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
"""

# layer = tf.keras.layers.Dense(100)
# layer(tf.zeros([10, 5]))
# print(layer.kernel.shape)
