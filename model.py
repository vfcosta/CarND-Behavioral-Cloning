import PIL
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Input, Flatten, Dense, Activation, Dropout, ELU
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing import image as kimage

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 50, "The batch size.")
flags.DEFINE_float('scale', 0.25, "Image scale.")
flags.DEFINE_string('data_dir', 'data', "Data dir.")

vertical_crop = (np.array([60, 135]) * FLAGS.scale).astype(int)
flip_threshold = 0.01


def crop_image(image_array):
    return image_array[vertical_crop[0]:vertical_crop[1], :, :]


def resize_image(image):
    """Resize image using ANTIALIAS algoithm for better performance in downscale"""
    return image.resize((int(320*FLAGS.scale), int(160*FLAGS.scale)), resample=PIL.Image.ANTIALIAS)


def normalize_image(image_array):
    """Convert image from [0, 255] to [-1, 1]"""
    return image_array / 128 - 1


def preprocess(image, normalize=True, flip=False):
    """Resize, crop, flip and normalize image"""
    image = resize_image(image)
    image_array = crop_image(np.asarray(image))
    if flip: image_array = np.fliplr(image_array)
    if normalize: image_array = normalize_image(image_array)
    return image_array


def calculate_steering(steering, image_label, flip=False):
    """Calculate steering angle based on image position"""
    steering_factor = {'center': 0, 'right': -0.25, 'left': 0.25}
    steering += steering_factor[image_label]
    if flip: steering *= -1
    return min(max(steering, -1), 1)  # keep steering in the range [-1, 1]


def generate_single(row, image_label='center', flip=False):
    image_path = row[image_label].strip()
    if not image_path.startswith('/'): image_path = FLAGS.data_dir + "/" + image_path
    image = kimage.load_img(image_path)
    steering = calculate_steering(row['steering'], image_label, flip=flip)
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
                if flip_threshold and image_label != 'center' and abs(row['steering']) <= flip_threshold: continue
                for flip in [False, True]:
                    # do not flip images with smaller steering angles
                    if flip_threshold and flip and abs(row['steering']) <= flip_threshold: continue
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
    out = Dropout(0.4)(out)
    out = Activation('relu')(out)

    out = Dense(50)(out)
    out = Dropout(0.4)(out)
    out = Activation('relu')(out)

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
    flip_len = len(data[abs(data['steering']) > flip_threshold]) if flip_threshold else data['steering']
    return len(data) + (2*positions - 1) * flip_len


def save_model(model):
    # save data
    print("saving model...")
    model.save_weights("model.h5")
    with open("model.json", 'w') as out:
        out.write(model.to_json())


def main(_):
    driving_log = pd.read_csv(FLAGS.data_dir + '/driving_log.csv', usecols=[0, 1, 2, 3])

    model = create_model()
    # split train/validation
    train_log, validation_log = train_test_split(driving_log, test_size=0.2)

    # select images to use in training
    image_labels = ['center', 'right', 'left']

    # calculate the length of training and validation data
    train_len = calculate_data_len(train_log, positions=len(image_labels))
    validation_len = calculate_data_len(validation_log, positions=len(image_labels))

    # train model
    try:
        model.fit_generator(generator=generate_data(train_log, image_labels=image_labels),
                            validation_data=generate_data(validation_log, image_labels=image_labels),
                            nb_epoch=FLAGS.epochs, samples_per_epoch=train_len, nb_val_samples=validation_len)
        save = "y" # force save when finish
    except KeyboardInterrupt:
        # option to save even when it's interrupted
        save = input("\n\nStopped! Do you want to save? <y/n>")

    if save == "y": save_model(model)
    print("Finished!")

# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
