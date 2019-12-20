#!/usr/bin/env python
# https://www.freecodecamp.org/news/how-to-train-a-core-ml-model-with-turi-create-to-classify-dog-breeds-bc1d0fa108b/

import turicreate as tc
import os

data = tc.image_analysis.load_images("Images/")

data["label"] = data["path"].apply(lambda path: os.path.basename(os.path.dirname(path)))

data.save("dog_classifier.sframe")

data = tc.SFrame("dog_classifier.sframe")

testing, training = data.random_split(.8)

classifier = tc.image_classifier.create(testing, target="label", model="resnet-50")

testing = classifier.evaluate(training)

print testing["accuracy"]

classifier.save("dog_classifier.model")

classifier.export_coreml("dog_classifier.mlmodel")

print "Trained model successfully. Happy classifying."
