import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class Estimator(Model):
    def __init__(self, num_actions):
        super(Estimator, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), 4, input_shape=(84, 84, 4), activation='relu', padding='same')
        self.conv2 = Conv2D(64, (4, 4), 2, activation='relu', padding='same')
        self.conv3 = Conv2D(64, (3, 3), 1, activation='relu', padding='same')
        self.flatten = Flatten()
        self.d1 = Dense(512)
        self.d2 = Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

    def predict(self, inputs):
        return self.call(inputs)

