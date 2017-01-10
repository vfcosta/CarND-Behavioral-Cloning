import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image as kimage
from keras.optimizers import Adam
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 50, "The batch size.")
flags.DEFINE_float('scale', 0.25, "Image scale.")
flags.DEFINE_string('data_dir', 'data', "Data dir.")


def preprocess(image):
    """Convert image from [0, 255] to [-1, 1]"""
    image = cv2.resize(image, (int(320*FLAGS.scale), int(160*FLAGS.scale)), interpolation=cv2.INTER_AREA)
    return image/128 - 1


def generate_data(data, size=FLAGS.batch_size):
    """Generator for data stored in driving log to be used in fit_generator from keras"""
    while True:
        images = []
        targets = []
        index = 0
        for _, row in data.iterrows():
            index += 1
            image = kimage.img_to_array(kimage.load_img(FLAGS.data_dir + "/" + row['center'], target_size=(int(160*FLAGS.scale), int(320*FLAGS.scale))))
            images.append(preprocess(image))
            targets.append(row['steering'])
            if index % size == 0 or index == len(data):
                yield (np.array(images), np.array(targets))
                images = []
                targets = []


def main(_):
    driving_log = pd.read_csv(FLAGS.data_dir + '/driving_log.csv', usecols=[0, 3])
    print(driving_log.head(4))

    # define input layer
    inp = Input(shape=(int(160*FLAGS.scale), int(320*FLAGS.scale), 3))
    # out = BatchNormalization(axis=3)(inp)
    # convolution layers
    out = Convolution2D(32, 5, 5)(inp)
    out = MaxPooling2D()(out)
    out = Activation('relu')(out)
    out = Convolution2D(64, 5, 5)(out)
    out = MaxPooling2D()(out)
    out = Activation('relu')(out)
    out = Convolution2D(128, 3, 3)(out)
    out = MaxPooling2D()(out)
    out = Activation('relu')(out)

    # Fully connected layers
    out = Flatten()(out)
    out = Dense(256)(out)
    out = Activation('relu')(out)
    out = Dense(64)(out)
    out = Activation('relu')(out)
    out = Dense(1)(out)

    # Define model
    model = Model(inp, out)
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # split train/validation
    train_log, validation_log = train_test_split(driving_log, test_size=0.2)

    # train model
    model.fit_generator(generator=generate_data(train_log), validation_data=generate_data(validation_log),
                        nb_epoch=FLAGS.epochs, samples_per_epoch=len(train_log), nb_val_samples=len(validation_log))

    # save data
    model.save_weights("model.h5")
    with open("model.json", 'w') as out:
        out.write(model.to_json())
    return 0

# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
