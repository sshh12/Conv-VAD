"""
Train a Conv-VAD model.

$ python model/train.py --data_path data --epochs 25
"""
import numpy as np
import click
import glob
import os


from keras.models import Model
from keras.regularizers import *
from keras.optimizers import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *


SHAPE = (400, 126, 1)


def get_model():
    """
    Create the VAD model architecture.
    """
    inp = Input(shape=SHAPE)

    x = Conv2D(64, (9, 9))(inp)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (5, 5))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(256, (3, 3))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = LeakyReLU()(x)

    x = Dense(1, activation='sigmoid')(x)
    out = x

    model = Model(inputs=inp, outputs=out)
    model.compile(Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])

    return model


@click.command()
@click.option('--data_path',
              required=True,
              help='Where training examples are saved.',
              type=click.Path())
@click.option('--model_path',
              required=True,
              help='Where to save trained model.',
              default='vad_best.h5',
              type=click.Path())
@click.option('--epochs',
              required=False,
              default=20,
              type=int)
@click.option('--batch_size',
              required=False,
              default=32,
              type=int)
def train(data_path=None, model_path=None, epochs=None, batch_size=None):

    X, Y = [], []

    print('Loading data...', end='')
    for fn in glob.iglob(os.path.join(data_path, '*.npy')):
        ary = np.load(fn).reshape(*SHAPE)
        X.append(ary)
        Y.append(1 if 'voice' in fn else 0)
    X = np.array(X)
    Y = np.array(Y)

    shuffle_idxs = np.random.permutation(X.shape[0])
    X = X[shuffle_idxs]
    Y = Y[shuffle_idxs]
    print('done.')

    print('X stats ->', X.mean(), X.std(), X.shape)
    print('Y stats ->', Y.mean(), Y.std(), Y.shape)

    # Precomputed Normalization
    X = (X - 0.643) / 0.094

    model = get_model()
    model.fit(X, Y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              callbacks=[ModelCheckpoint(filepath=model_path, save_best_only=True)])


if __name__ == '__main__':
    train()
