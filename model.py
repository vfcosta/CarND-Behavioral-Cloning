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
flags.DEFINE_integer('batch_size', 100, "The batch size.")
flags.DEFINE_float('scale', 0.25, "Image scale.")
flags.DEFINE_string('data_dir', 'data_sdc', "Data dir.")

vertical_crop = (np.array([60, 135]) * FLAGS.scale).astype(int)
flip_threshold = 0.01


def crop(image_array):
    return image_array[vertical_crop[0]:vertical_crop[1], :, :]


def preprocess(image, normalize=True, flip=False):
    """Rescale, crop, and convert image from [0, 255] to [-1, 1]"""
    image = image.resize((int(320*FLAGS.scale), int(160*FLAGS.scale)))
    image_array = np.asarray(image)
    image_array = crop(image_array)
    if flip: image_array = np.fliplr(image_array)
    if normalize: image_array = image_array / 128 - 1
    return image_array


def generate_single(row, image_label='center', flip=False):
    steering_factor = {'center': 0, 'right': -0.25, 'left': 0.25}
    image = kimage.load_img(FLAGS.data_dir + "/" + row[image_label].strip())
    steering = row['steering'] + steering_factor[image_label]
    if flip: steering *= -1
    return preprocess(image, flip=flip), steering


def generate_data(data, size=FLAGS.batch_size, image_labels=['center', 'right', 'left']):
    """Generator for data stored in driving log to be used in fit_generator from keras"""
    while True:
        images = []
        targets = []
        index = 0
        for _, row in data.iterrows():
            index += 1
            for image_label in image_labels:
                for flip in [False, True]:
                    # skip images with smaller steering angles
                    if flip and abs(row['steering']) <= flip_threshold: continue
                    image, target = generate_single(row, image_label=image_label, flip=flip)
                    images.append(image)
                    targets.append(target)
            if index % size == 0 or index == len(data):
                yield (np.array(images), np.array(targets))
                images = []
                targets = []


def create_model():
    # define input layer
    inp = Input(shape=(vertical_crop[1] - vertical_crop[0], int(320 * FLAGS.scale), 3))

    # out = BatchNormalization(axis=3)(inp)

    # convolution layers
    out = Convolution2D(16, 3, 3)(inp)
    out = MaxPooling2D()(out)
    out = Activation('relu')(out)

    out = Convolution2D(64, 3, 3)(out)
    out = MaxPooling2D()(out)
    out = Activation('relu')(out)

    # Fully connected layers
    out = Flatten()(out)

    out = Dense(256)(out)
    out = Dropout(0.3)(out)
    out = Activation('relu')(out)

    # out = Dense(100)(out)
    # out = Dropout(0.5)(out)
    # out = Activation('relu')(out)

    # Output layer
    out = Dense(1)(out)

    # Define model
    model = Model(inp, out)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model


def calculate_data_len(data, positions=3):
    """
    Calculate the length of data based on augmentation strategies
    positions = 3 # center, right and left images
    """
    flip_len = len(data[abs(data['steering']) > flip_threshold])
    return positions * (len(data) + flip_len)


def main(_):
    driving_log = pd.read_csv(FLAGS.data_dir + '/driving_log.csv', usecols=[0, 1, 2, 3])

    model = create_model()
    # split train/validation
    train_log, validation_log = train_test_split(driving_log, test_size=0.2)

    # calculate the length of training and validation data
    train_len = calculate_data_len(train_log)
    validation_len = calculate_data_len(validation_log)

    # train model
    model.fit_generator(generator=generate_data(train_log), validation_data=generate_data(validation_log),
                        nb_epoch=FLAGS.epochs, samples_per_epoch=train_len, nb_val_samples=validation_len)

    # save data
    model.save_weights("model.h5")
    with open("model.json", 'w') as out:
        out.write(model.to_json())
    print("Finished!")

# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
