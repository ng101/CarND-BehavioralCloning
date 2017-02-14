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
from PIL import Image
import math

import tensorflow as tf
tf.python.control_flow_ops = tf

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0,0))))
    # Conv
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    # Conv
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    # Conv
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    # Conv
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # Conv
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    
    # Flatten
    model.add(Flatten())
    model.add(Dropout(0.5))
    # Dense
    model.add(Dense(128, activation='relu'))
    # Dense
    model.add(Dense(64, activation='relu'))
    # Dense
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    model.summary()
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
            flip = batch[i, 2]
            image_file = batch[i, 0].strip()
            image = Image.open(os.path.join(root, image_file))
            image_array = np.asarray(image)
            X[i, :, :, :] = image_array if 0 == flip else np.fliplr(image_array)
            #(mpimg.imread(os.path.join(root, image_file)).astype(float))
            Y[i] = batch[i, 1] if 0 == flip else -1.0 * batch[i, 1]
        yield (X, Y)

def train():
    df = pd.read_csv('../tr/driving_log.csv')
    dfc = df[['center', 'steering']].rename(columns={'center':'image'})
    dfl = df[['left', 'steering']].rename(columns={'left':'image'})
    dfl['steering'] = dfl['steering'] + 0.2
    dfr = df[['right', 'steering']].rename(columns={'right':'image'})
    dfr['steering'] = dfr['steering'] - 0.2
    dfn = pd.concat([dfc, dfl, dfr])
    df = dfn.dropna() # beta simulator doesn't give left/right images
    df['flip'] = 0
    df_flipped = df.copy()
    df_flipped['flip'] = 1

    df = pd.concat([df, df_flipped])

    print(df.head(1))
    print(df['steering'].quantile([0.01,0.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,0.99]))
    print(df['steering'].describe())
    df = shuffle(df)
    train, valid = train_test_split(df, test_size = 0.20)
    model = get_model()
    batch_size = 256
    root = '../tr/'
    model.fit_generator(
            gen(train, batch_size, root),
            samples_per_epoch = (len(train) // batch_size) * batch_size,
            nb_epoch=30,
            validation_data=gen(valid, batch_size, root),
            nb_val_samples=len(valid)
            )
    model.save('model.h5')

if __name__ == '__main__':
    train()
