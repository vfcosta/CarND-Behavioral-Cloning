import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from scipy.misc import imread
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def preprocess(image):
    """Convert image from [0, 255] to [-0.5, 0.5]"""
    return image/255 - 0.5


def generate_data(driving_log):
    """Generator for data stored in driving log to be used in fit_generator from keras"""
    size = 20
    while True:
        images = []
        targets = []
        for index, row in driving_log.iterrows():
            image = imread(row['center_image']).astype(np.float32)
            images.append(preprocess(image))
            targets.append(row['steering_angle'])
            if index % size == 0:
                yield (np.array(images), np.array(targets))
                images = []
                targets = []

        if len(images) < 50:
            yield (np.array(images), np.array(targets))


def main(_):
    driving_log = pd.read_csv('data/driving_log.csv', header=None, usecols=[0, 3], names=['center_image', 'steering_angle'])
    print(driving_log.ix[4])
    print(driving_log.head(4))
    # print(next(generate_data(driving_log)))

    # define model
    inp = Input(shape=(160, 320, 3))
    # convolution layers
    out = Convolution2D(32, 3, 3)(inp)
    out = MaxPooling2D()(out)
    out = Convolution2D(64, 3, 3)(out)
    out = MaxPooling2D()(out)
    out = Convolution2D(128, 3, 3)(out)
    out = MaxPooling2D()(out)
    out = Convolution2D(64, 3, 3)(out)
    out = MaxPooling2D()(out)

    # Fully connected layers
    out = Flatten()(out)
    out = Dense(32)(out)
    out = Dense(1)(out)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # train model
    model.fit_generator(generate_data(driving_log), nb_epoch=FLAGS.epochs, samples_per_epoch=len(driving_log))


# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
