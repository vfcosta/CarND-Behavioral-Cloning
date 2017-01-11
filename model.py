import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Flatten, Dense, Activation, Dropout, ELU
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image as kimage
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 50, "The batch size.")
flags.DEFINE_float('scale', 0.5, "Image scale.")
flags.DEFINE_string('data_dir', 'data_sdc', "Data dir.")


def preprocess(image):
    """Rescale, crop, and convert image from [0, 255] to [-1, 1]"""
    image = image.resize((int(160*FLAGS.scale), int(320*FLAGS.scale)))
    image_array = np.asarray(image)
    # image = cv2.resize(image, (int(320*FLAGS.scale), int(160*FLAGS.scale)), interpolation=cv2.INTER_AREA)
    image_array = image_array[int(55*FLAGS.scale):int(135*FLAGS.scale), :, :]
    return image_array/128 - 1


def generate_data(data, size=FLAGS.batch_size):
    """Generator for data stored in driving log to be used in fit_generator from keras"""
    while True:
        images = []
        targets = []
        index = 0
        for _, row in data.iterrows():
            index += 1
            image = kimage.load_img(FLAGS.data_dir + "/" + row['center'])
            images.append(preprocess(image))
            targets.append(row['steering'])
            if index % size == 0 or index == len(data):
                yield (np.array(images), np.array(targets))
                images = []
                targets = []


def main(_):
    driving_log = pd.read_csv(FLAGS.data_dir + '/driving_log.csv', usecols=[0, 1, 2, 3])
    print(driving_log.head(4))

    # define input layer
    # inp = Input(shape=(int(160*FLAGS.scale), int(320*FLAGS.scale), 3))
    inp = Input(shape=(40, 80, 3))
    # out = BatchNormalization(axis=3)(inp)
    # convolution layers
    out = Convolution2D(16, 5, 5, subsample=(2, 2))(inp)
    # out = MaxPooling2D()(out)
    # out = Activation('relu')(out)
    out = ELU()(out)

    out = Convolution2D(32, 5, 5, subsample=(2, 2))(out)
    # out = MaxPooling2D()(out)
    # out = Activation('relu')(out)
    out = ELU()(out)

    out = Convolution2D(64, 3, 3)(out)
    # out = MaxPooling2D()(out)
    # out = Activation('relu')(out)
    out = ELU()(out)

    # Fully connected layers
    out = Flatten()(out)

    out = Dense(512)(out)
    out = Dropout(0.5)(out)
    out = ELU()(out)

    out = Dense(100)(out)
    out = Dropout(0.5)(out)
    out = ELU()(out)

    # Output layer
    out = Dense(1)(out)

    # Define model
    model = Model(inp, out)
    model.compile(optimizer="adam", loss='mean_squared_error')

    # split train/validation
    train_log, validation_log = train_test_split(driving_log, test_size=0.2)

    # train model
    model.fit_generator(generator=generate_data(train_log), validation_data=generate_data(validation_log),
                        nb_epoch=FLAGS.epochs, samples_per_epoch=len(train_log), nb_val_samples=len(validation_log))

    # save data
    model.save_weights("model.h5")
    with open("model.json", 'w') as out:
        out.write(model.to_json())
    print("Finished!")

# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
