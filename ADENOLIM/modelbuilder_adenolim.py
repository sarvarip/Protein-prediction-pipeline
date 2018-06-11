import numpy as np
import config as C

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

np.random.seed(C.random_seed)

def build_sequential_model(rate, shape):
    model = Sequential()

    model.add(Dropout(0.2, input_shape=(shape,)))

    model.add(Dense(64, activation="linear", kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(rate))

    model.add(Dense(32, activation="linear", kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(rate))

    model.add(Dense(16, activation="linear", kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(rate))

    model.add(Dense(activation="sigmoid", units=1))

    optim = Adam(lr=0.01, beta_1=0.95)

    model.compile(loss='binary_crossentropy',
                    optimizer=optim,
                    metrics=['accuracy'])
    return model


def fit_model_batch(model, x, y, num_epoch=None):
    if num_epoch is None:
        num_epoc = 1000
    es = [EarlyStopping(monitor='loss', min_delta=0, patience=200, verbose=0, mode='auto')]
    model.fit(x, y, epochs=num_epoch, batch_size=x.shape[0], callbacks = es) #full batch size
    return model
