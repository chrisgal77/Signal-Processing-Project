from model import get_model
from dataset import get_data

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LEARNING_RATE = 1e3
BATCH_SIZE = 16
HEIGHT = 299
WIDTH = 299
EPOCHS = 15

ds_train, ds_validation = get_data(HEIGHT, WIDTH, BATCH_SIZE)
model = get_model((HEIGHT, WIDTH, 3))

model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(ds_train,
          epochs=EPOCHS)
