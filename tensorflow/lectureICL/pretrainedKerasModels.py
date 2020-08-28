#!/usr/bin/env python

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd

model = ResNet50(weights='imagenet')
model.summary()

lemon_img = load_img('data/lemon.jpg', target_size=(224, 224))
viaduct_img = load_img('data/viaduct.jpg', target_size=(224, 224))
water_tower_img = load_img('data/water_tower.jpg', target_size=(224, 224))

def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...]
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x), top=5)
    top_preds = pd.DataFrame(columns=['prediction', 'probability'],
                             index=np.arange(5)+1)
    for i in range(5):
        top_preds.loc[i+1, 'prediction'] = preds[0][i][1]
        top_preds.loc[i+1, 'probability'] = preds[0][i][2]
    return top_preds

#lemon_img
print("Lemon image probabilities")
print(get_top_5_predictions(lemon_img))

#viaduct_img
print("Viaduct image probabilities")
print(get_top_5_predictions(viaduct_img))

#water_tower_img
print("Water tower image probabilities")
print(get_top_5_predictions(water_tower_img))
