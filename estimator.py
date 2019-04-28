import tensorflow as tf
import os
from tensorflow.keras import Model


class Estimator(Model):
    def __init__(self, num_actions):
        super(Estimator, self).__init__()
        # self.input = tf.keras.layers.Input(shape=(84, 84, 3))
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4,
                                            input_shape=(84, 84, 3),
                                            activation='relu',
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros')
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2,
                                            activation='relu',
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1,
                                            activation='relu',
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(512)
        self.d2 = tf.keras.layers.Dense(num_actions)


    @tf.function  #
    def call(self, inputs):
        # inputs = tf.multiply(inputs, 1.0) / 255.0
        # inputs = self.input(input_data)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        outputs = self.d2(x)
        return outputs
