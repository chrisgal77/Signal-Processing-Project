import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

def get_model(input_shape):
    model_path = 'https://tfhub.dev/tensorflow/efficientnet/b4/classification/1'

    print('=> Downloading model')
    base = hub.KerasLayer(model_path, input_shape=input_shape)
    model = keras.Sequential(
        [
            base,
            layers.Dense(128, activation="relu"),
            layers.Dense(5),
        ]
    )
    print('=> Downloaded model')
    return model
