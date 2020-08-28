#!/usr/bin/env python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# Build the Sequential convolutional neural network model
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(1, 28,28), data_format="channels_first"),
    MaxPooling2D((3,3), data_format="channels_first"),
    Flatten(),
    Dense(10, activation="softmax")
])

model.add(Dense(64,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation='relu'),)

model.add(Dense(8,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation='relu'))

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))

model.summary()

fig, axes = plt.subplots(5, 2, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Filter out the pooling and flatten layers, that don't have any weights
weight_layers = [layer for layer in model.layers if len(layer.weights) > 0]

for i, layer in enumerate(weight_layers):
    for j in [0, 1]:
        axes[i, j].hist(layer.weights[j].numpy().flatten(), align='left')
        axes[i, j].set_title(layer.weights[j].name)
