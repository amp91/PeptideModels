"""
Author: Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""

import keras
import numpy as np
from keras.layers import (Dense, Dropout, Conv1D,
                          BatchNormalization, MaxPool1D, Flatten, Input)
from keras.models import Sequential, Model
from keras.models import load_model
from keras.regularizers import l2
from pathlib import Path
from typing import List, Tuple


def build_model(in_shape: Tuple) -> Model:
    """
    Build multi-task regression model
    :param in_shape: shape of the input
    :return: compiled model
    """

    def base_forward_model():
        base_model = Sequential()
        base_model.add(Conv1D(256, kernel_size=3, padding='valid',
                              activation='relu',
                              kernel_regularizer=l2(0.01),
                              name='conv1'))
        base_model.add(BatchNormalization(name='batch_normalization1'))
        base_model.add(MaxPool1D(pool_size=2, name='max_pool1'))
        base_model.add(Dropout(0.5, name='dropout1'))
        base_model.add(Conv1D(512, kernel_size=3, padding='valid',
                              activation='relu',
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01), name='conv2'))
        base_model.add(BatchNormalization(name='batch_normalization2'))
        base_model.add(MaxPool1D(pool_size=2, name='max_pool2'))
        base_model.add(Dropout(0.5, name='dropout2'))
        base_model.add(Conv1D(128, kernel_size=3, padding='same',
                              activation='relu',
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01), name='conv3'))
        base_model.add(BatchNormalization(name='batch_normalization3'))
        base_model.add(MaxPool1D(pool_size=2, name='max_pool3'))
        base_model.add(Dropout(0.5, name='dropout3'))

        base_model.add(Flatten(name='flatten'))
        base_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01), name='dense1'))
        base_model.add(Dense(64, activation='relu', name='dense2'))

        return base_model

    input_seqs = Input(shape=in_shape)
    model = base_forward_model()
    encoded_seq = model(input_seqs)

    output1 = Dense(1, name='R1_output')(encoded_seq)
    output2 = Dense(1, name='R2_output')(encoded_seq)

    multi_model = Model(inputs=input_seqs, outputs=[output1, output2])

    multi_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'],
                        loss_weights=[0.5, 0.5])

    return multi_model


def get_callbacks():
    return [keras.callbacks.EarlyStopping(monitor='val_loss', patience=75)]


def fit_model(model: Model,
              x_train: np.ndarray,
              y_train: List,
              max_epochs: int,
              val_data: Tuple[np.ndarray, List]):
    """
    Fit model
    :param model: compiled model
    :param x_train: training batch
    :param y_train: true target(s) values
    :param max_epochs: maximal number of training epochs
    :param val_data: validation data
    :return: training history
    """
    history = model.fit(x_train, y_train,
                        epochs=max_epochs,
                        batch_size=25, callbacks=get_callbacks(),
                        validation_data=val_data, verbose=0)
    return history


def get_models(path_models: Path) -> List[Model]:
    """
    Reads saved models
    :param path_models: path to the directory
    with trained models
    :return: list of model instances
    """
    infers = []
    for i, file in enumerate(sorted(path_models.iterdir())):
        print('loading model:', str(file))
        model = load_model(str(file))
        infers.append(model)
    return infers


def get_predictions(path_to_models: Path,
                    X_test: np.ndarray) -> np.ndarray:
    """
    Loads saved models and predict for batch of sequences

    :param path_to_models: path to pre-trained models
    :param X_test: test data
    dim = [num_test_seqs x len_seq x encoding_dim]
    :return: array with predictions
    dim = [num_models x num_targets x num_test_seqs]
    """
    print(f'Number of models in the ensemble:'
          f'{len(sorted(path_to_models.iterdir()))}')
    y_predicted = []
    for f_path in sorted(path_to_models.iterdir()):
        print(f'loading model:{str(f_path)}')
        model = load_model(str(f_path))
        y_hat_current = model.predict(X_test)
        y_predicted.append(y_hat_current)
        del model

    return np.asarray(y_predicted)
