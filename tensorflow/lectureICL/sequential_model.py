#!/usr/bin/env python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax

model = Sequential([
    Flatten(input_shape=(20,20), name="layer_input"),
    Dense(16, activation="relu", name="layer_1"),
    Dense(16, activation="relu", name="layer_2"),
    Dense(10, name="layer_3"),
    Softmax(name="layer_output")
])

model.summary()
