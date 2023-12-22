# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools import selectFeatures, getTarget, saveSubmission


def kerasNetwork(X_set, X_test, y_set, model_structure = None, seed_num=1, batch_size=128, epochs=1000, lr=0.0016, decay=0.004, patience=150, verbose=True, export=False, name="NeuralNetwork"):
    """
    Builds, trains, and evaluates a Keras neural network.
    Note : Parameter unpacking can be used (*) to pass training and test sets to the function. As selectFeatures retuns a tuple (X_set, X_test), *selectFeatures returns X_set, X_test

    Parameters
    ----------
    X_set : pandas.DataFrame or numpy.ndarray
        The training features.
    X_test : pandas.DataFrame or numpy.ndarray
        The test features.
    y_set : pandas.Series or numpy.ndarray
        The training targets.
    model_structure : list, optional
        The structure of the model as a list of Keras layers. If None, a default structure is used.
    seed_num : int, optional
        The seed for random number generation, for reproducibility.
    batch_size : int, optional
        The batch size for training.
    epochs : int, optional
        The number of epochs to train for.
    lr : float, optional
        The learning rate for the AdamW optimizer.
    decay : float, optional
        The decay rate for the AdamW optimizer.
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped.
    verbose : bool, optional
        Whether to print verbose messages during training.
    export : bool, optional
        Whether to export the final predictions.
    name : str, optional
        The name of the model, used when exporting the predictions under the folder "Submissions/KerasNetwork".

    Returns
    -------
    tuple
        A tuple containing the trained model, the training history, the validation loss and predictions for the test set.
    """
    
    # Reproducibility
    keras.utils.set_random_seed(seed_num)

    # Splitting test and train
    X_train, X_val, y_train, y_val = train_test_split(
        X_set,
        y_set,
        test_size=0.20,
        random_state=seed_num
    )

    # Standardize data
    Xscaler = StandardScaler()
    X_std = Xscaler.fit_transform(X_train)
    X_val_std = Xscaler.transform(X_val)
    X_test_std = Xscaler.transform(X_test)
    Yscaler = StandardScaler()
    y_std = Yscaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_std = Yscaler.transform(y_val.values.reshape(-1, 1))


    if model_structure == None:
        model_structure = [
            layers.Dense(416, activation='relu'),
            layers.Dense(352, activation='relu'),
            layers.Dense(288, activation='relu'),
            layers.Dense(416, activation='relu'),
            layers.Dense(1)
        ]

    # Setting up the model
    def build_model():
        model_structure.insert(0, keras.Input(shape=X_set.shape[1]))
        
        model = keras.Sequential(model_structure)

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=decay),
            loss="mean_squared_error"
        )
        
        return model

    model = build_model()
    model.summary() if verbose else None
    history = model.fit(
        X_std,
        y_std,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_std, y_val_std),
        verbose=verbose,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    )

    # Predictions for the validation set
    y_pred_val = model.predict(X_val_std, verbose=verbose)
    y_pred_val = Yscaler.inverse_transform(y_pred_val).reshape(-1)

    y_pred_train = model.predict(X_std, verbose=verbose)
    y_pred_train = Yscaler.inverse_transform(y_pred_train).reshape(-1)

    # Compute mean squared error
    val_loss = mean_squared_error(y_val, y_pred_val)
    loss = mean_squared_error(y_train, y_pred_train)

    # Inverse transform the losses and print
    print(f"Train loss : {loss}, Validation loss: {val_loss}") if verbose else None

    # Predictions for the test set
    y_pred = model.predict(X_test_std, verbose=verbose)
    y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

    # Transforming to array and saving
    y_pred = np.array(y_pred)
    if export:
        saveSubmission(y_pred, f'KerasNetworks/{name}')

    return model, history, val_loss, y_pred