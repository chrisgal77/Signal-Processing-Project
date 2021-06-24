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
    ds_train, ds_validation = get_data(HEIGHT, WIDTH, batch_size, data_path)
    model = get_model((HEIGHT, WIDTH, 3))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    fn_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    train_writer = tf.summary.create_file_writer('logs/train/')
    val_writer = tf.summary.create_file_writer('logs/val/')

    print('=> Learning has started')
    for epoch in range(epochs):
        for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss = fn_loss(y_batch, y_pred)
                
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y_batch, y_pred)

        with train_writer.as_default():
            tf.summary.scalar('Loss', loss, step=epoch)
            tf.summary.scalar('Accuracy', acc_metric.result(), step=epoch)
            
        train_acc = acc_metric.result()
            
        for batch_idx, (x_batch, y_batch) in enumerate(ds_validation):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=False)
                loss = fn_loss(y_batch, y_pred)

            acc_metric.update_state(y_batch, y_pred)
            
        with val_writer.as_default():
            tf.summary.scalar('Loss', loss, step=epoch)
            tf.summary.scalar('Accuracy', acc_metric.result(), step=epoch)

        val_acc = acc_metric.result()
        acc_metric.reset_states()  
    
        print(f'train accuracy: {train_acc} val accuracy: {val_acc}')


if __name__ == "__main__":
    args = get_args()
    train(
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        epochs = args.epochs,
        data_path = args.data_path
    )
