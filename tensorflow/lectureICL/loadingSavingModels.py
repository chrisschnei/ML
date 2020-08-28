#!/usr/bin/env python

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Use smaller subset -- speeds things up
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])

def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))

def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_new_model()
model.summary()

get_test_accuracy(model,x_test,y_test)

checkpoint_path = 'model_checkpoints/checkpoint'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, frequency='epoch',
                            save_weights_only=True,
                            verbose=1)

model.fit(x=x_train, y=y_train, epochs=3, callbacks=[checkpoint])

get_test_accuracy(model, x_test, y_test)

model= get_new_model()
get_test_accuracy(model, x_test, y_test)

model.load_weights(checkpoint_path)
get_test_accuracy(model, x_test, y_test)

checkpoint_5000_path = 'model_checkpoints_5000/checkpoint_{batch:04d}'
checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path, save_weights_only=True, save_freq=5000, verbose=1)

model = get_new_model()
model.fit(x=x_train, y=y_train, epochs=3, validation_data=(x_test, y_test), batch_size=10, callbacks=[checkpoint_5000])

x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

model = get_new_model()

checkpoint_best_path = 'model_checkpoints_best/checkpoint'
checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_weights_only=True, save_freq='epoch', monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test), batch_size=10, callbacks=[checkpoint_best], verbose=1)

df = pd.DataFrame(history.history)
df.plot(y=['accuracy', 'val_accuracy'])

new_model = get_new_model()
new_model.load_weights(checkpoint_best_path)
get_test_accuracy(new_model, x_test, y_test)

# save entire model
checkpoint_path = 'model_checkpoints'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, frequency='epoch', verbose=1)

model = get_new_model()
model.fit(x=x_train, y=y_train, epochs=3, callbacks=[checkpoint])

get_test_accuracy(model, x_test, y_test)

del model

model = load_model(checkpoint_path)
get_test_accuracy(model, x_test, y_test)

model.save('my_model.h5')

del model

model = load_model('my_model.h5')
get_test_accuracy(model, x_test, y_test)
