#!/usr/bin/env python

import os
import matplotlib.image as mpimg
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Sequential
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import tensorflow as tf
tf.python.control_flow_ops = tf

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
    #model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(1,1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(36, 5, 5, subsample=(1,1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(48, 5, 5, subsample=(1,1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    return model

def gen(df, batch_size, root=''):
    df = df.as_matrix()
    start = 0
    while True:
        batch = df[start:start+batch_size, :]
        start = (start + batch_size) % len(df)
        if (len(df) - start) < batch_size:
            start = 0
        X = np.ndarray(shape=(len(batch), 160, 320, 3), dtype=np.float32)
        Y = np.zeros((len(batch)))
        for i in range(len(batch)):
            image_file = batch[i, 0].strip()
            X[i, :, :, :] = (mpimg.imread(os.path.join(root, image_file)).astype(float))
            Y[i] = batch[i, 1]
        yield (X, Y)

def train():
    df = pd.read_csv('../training_data_2/driving_log.csv')
    dfc = df[['center', 'steering']].rename(columns={'center':'image'})
    dfl = df[['left', 'steering']].rename(columns={'left':'image'})
    dfl['steering'] = dfl['steering'] + 0.2
    dfr = df[['right', 'steering']].rename(columns={'right':'image'})
    dfr['steering'] = dfr['steering'] - 0.2
    dfn = pd.concat([dfc, dfl, dfr])
    df = dfn.dropna() # beta simulator doesn't give left/right images

    df = shuffle(df)
    train, valid = train_test_split(df, test_size = 0.33)
    model = get_model()
    batch_size = 128
    root = '../training_data_2/'
    model.fit_generator(
            gen(train, batch_size, root),
            samples_per_epoch = (len(train) // batch_size) * batch_size,
            nb_epoch=20,
            validation_data=gen(valid, batch_size, root),
            nb_val_samples=len(valid)
            )
    model.save('model.h5')

if __name__ == '__main__':
    train()
