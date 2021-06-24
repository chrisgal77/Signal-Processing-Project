from model import get_model
from dataset import get_data

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LEARNING_RATE = 1e3
BATCH_SIZE = 16
EPOCHS = 15

def get_args():
    parser = argparse.ArgumentParser(description='Train the model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr',
                        '--learningrate',
                        type=float,
                        help='Learning rate',
                        dest='learning_rate',
                        default=1e-3)
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        help='Batch size',
                        dest='batch_size',
                        default=64)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        help='Number of epochs',
                        dest='epochs',
                        default=10)
    parser.add_argument('-d',
                        '--datapath',
                        type=str,
                        help='Absolute path to the data',
                        dest='data_path')
    return parser.parse_args()

def train(learning_rate, batch_size, epochs, data_path):
    
    HEIGHT = 299
    WIDTH = 299
    print(type(data_path), data_path)
    ds_train, ds_validation = get_data(HEIGHT, WIDTH, BATCH_SIZE, data_path)
    model = get_model((HEIGHT, WIDTH, 3))

    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(ds_train,
            epochs=EPOCHS)


if __name__ == "__main__":
    args = get_args()
    train(
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        epochs = args.epochs,
        data_path = args.data_path
    )
