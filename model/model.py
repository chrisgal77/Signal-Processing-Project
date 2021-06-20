import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

model_path = 'https://tfhub.dev/tensorflow/efficientnet/b4/classification/1'

base = hub.KerasLayer(model_path, input_shape=(380, 380, 3))
model = keras.Sequential(
    [
        base,
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", 'f1_score'],
)
