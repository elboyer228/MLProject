# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np

import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools import selectFeatures, getTarget, saveSubmission


# Reproducibility
seed_num = 1
keras.utils.set_random_seed(seed_num)

# Model configuration
batch_size = 70
loss_function = mean_squared_error
no_epochs = 1000
verbosity = 1
patience = 150
num_folds = 5

# Load data
X_set, X_test = selectFeatures(Lab=True, mol=True, cddd=True, bestCddd=100)
y_set = getTarget()


# Normalize data
Xscaler = StandardScaler()
X_std = Xscaler.fit_transform(X_set)
X_test_std = Xscaler.transform(X_test)
Yscaler = StandardScaler()
y_std = Yscaler.fit_transform(y_set.values.reshape(-1, 1))

# Define per-fold score containers
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed_num)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X_std, y_std):

    # Define the model architecture
    model = keras.Sequential([
            keras.Input(shape=X_set.shape[1]),
            layers.Dense(416, activation='relu'),
            layers.Dense(352, activation='relu'),
            layers.Dense(288, activation='relu'),
            layers.Dense(416, activation='relu'),
            layers.Dense(1)
        ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate = 0.0016,
            weight_decay = 0.004
        ),
        loss=loss_function,
    )


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X_std[train], y_std[train],
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                callbacks=[EarlyStopping(monitor='val_loss', patience=patience)],
                validation_data=(X_std[test], y_std[test])
                )

    # Generate generalization metrics
    y_pred = model.predict(X_std[test], verbose = 0)
    y_pred = Yscaler.inverse_transform(y_pred)
    y_values = Yscaler.inverse_transform(y_std[test])
    
    loss_values = mean_squared_error(y_values, y_pred)
    loss = np.mean(loss_values)
    
    print(f'Score for fold {fold_no}: mse = {loss}')
    loss_per_fold.append(loss)

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')