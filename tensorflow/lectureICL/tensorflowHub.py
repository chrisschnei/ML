#!/usr/bin/env python

import tensorflow_hub as hub
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
model = Sequential([hub.KerasLayer(module_url)])
model.build(input_shape=[None, 160, 160, 3])
model.summary()

lemon_img = load_img("data/lemon.jpg", target_size=(160, 160))
viaduct_img = load_img("data/viaduct.jpg", target_size=(160, 160))
water_tower_img = load_img("data/water_tower.jpg", target_size=(160, 160))

with open('data/imagenet_categories.txt') as txt_file:
    categories = txt_file.read().splitlines()

def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...] / 255.0
    preds = model.predict(x)
    top_preds = pd.DataFrame(columns=['prediction'],
                             index=np.arange(5)+1)
    sorted_index = np.argsort(-preds[0])
    for i in range(5):
        ith_pred = categories[sorted_index[i]]
        top_preds.loc[i+1, 'prediction'] = ith_pred
            
    return top_preds

#lemon_img
print("Lemon image probabilities")
print(get_top_5_predictions(lemon_img))
