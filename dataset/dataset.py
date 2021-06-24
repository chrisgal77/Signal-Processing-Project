import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(height, width, batch_size, data_path):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(height, width),
        shuffle=True,
        seed=123,
        validation_split=0.8,
        subset="training",
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(height, width),
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset="validation",
    )

    return ds_train, ds_validation
