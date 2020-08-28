#!/usr/bin/env python

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()
data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

class TrainingCallback(Callback):
    def on_train_begin(self,logs=None):
        print("Starting training...")

    def on_epoch_begin(self,epoch,logs=None):
        print(f'Starting epoch {epoch}')

    def on_train_batch_begin(self,batch,logs=None):
        print(f'Training: Starting batch {batch}')

    def on_train_batch_end(self,batch,logs=None):
        print(f'Training: Finished batch {batch}')

    def on_epoch_end(self,epoch,logs=None):
        print(f'Finished epoch {epoch}')

    def on_train_end(self,logs=None):
        print('Finished training!')

class TestingCallback(Callback):
    def on_test_begin(self,logs=None):
        print("Starting testing...")

    def on_test_batch_begin(self,batch,logs=None):
        print(f'Testing: Starting batch {batch}')

    def on_test_batch_end(self,batch,logs=None):
        print(f'Testing: Finished batch {batch}')

    def on_test_end(self,logs=None):
        print('Finished testing!')

class PredictionCallback(Callback):
    def on_predict_begin(self,logs=None):
        print("Starting predicting...")

    def on_predict_batch_begin(self,batch,logs=None):
        print(f'Predicting: Starting batch {batch}')

    def on_predict_batch_end(self,batch,logs=None):
        print(f'Predicting: Finished batch {batch}')

    def on_predict_end(self,logs=None):
        print('Finished predicting!')

def get_regularised_model(wd, rate):
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(1)
    ])
    return model

model=get_regularised_model(1e-5, 0.3)

model.compile(optimizer="adam", loss="mse")

model.fit(train_data, train_targets, epochs=3, batch_size=128, verbose=False, callbacks=[TrainingCallback()])

model.evaluate(test_data, test_targets, verbose=False, callbacks=[TestingCallback()])

model.predict(test_data, verbose=False, callbacks=[PredictionCallback()])
