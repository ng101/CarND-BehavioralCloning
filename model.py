#!/usr/bin/env python

import os
import matplotlib.image as mpimg
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import tensorflow as tf
tf.python.control_flow_ops = tf

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model

def gen(df, root_dir, batch_size):
    df = df.as_matrix()
    start = 0
    while True:
        batch = df[start:start+batch_size, :]
        start = (start + batch_size) % len(df)
        X = np.ndarray(shape=(len(batch), 160, 320, 3), dtype=np.float32)
        Y = np.zeros((len(batch)))
        for i in range(len(batch)):
            image_file = batch[i, 0]
            X[i, :, :, :] = (mpimg.imread(os.path.join(root_dir, image_file)).astype(float))
            Y[i] = batch[i, 3]
        yield (X, Y)

def train():
    df = pd.read_csv('../data/driving_log_sample.csv')
    df = shuffle(df)
    train, valid = train_test_split(df, test_size = 0.33)
    model = get_model()
    model.fit_generator(
            gen(train, '../data/', 128),
            samples_per_epoch = len(train),
            nb_epoch=10,
            validation_data=gen(valid, '../data/', 128),
            nb_val_samples=len(valid)
            )
    model.save('model.h5')

if __name__ == '__main__':
    train()