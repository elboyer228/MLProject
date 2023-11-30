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
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools import selectFeatures, getTarget, plotHistory, saveSubmission



X_set, X_test = selectFeatures(Lab=True, mol=True)
y_set = getTarget()

# Reproducibility
seed_num = 42
np.random.seed(seed_num)
tf.random.set_seed(seed_num)


# Standardize data
Xscaler = StandardScaler()
X_std = Xscaler.fit_transform(X_set)
X_test_std = Xscaler.transform(X_test)
Yscaler = StandardScaler()
y_std = Yscaler.fit_transform(y_set.values.reshape(-1, 1))

 
# Setting up the model
def build_model():
    model = keras.Sequential([
        keras.Input(shape=X_set.shape[1]),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    return model


model = build_model()
model.summary()
history = model.fit(
    X_std,
    y_std,
    epochs=1000,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

# Predictions for the test set
y_pred = model.predict(X_test_std)
y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

# Transforming to array and saving
y_pred = np.array(y_pred)
# saveSubmission(y_pred, 'kerasNetwork')




######### Model tuning #########

def build_model_tunned(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=X_set.shape[1]))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1))
    
    
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model

tuner = RandomSearch(
    build_model_tunned,
    objective='val_mean_squared_error',
    max_trials=50,
    executions_per_trial=3,
    overwrite=True,
    directory='Hyperparameter_tuning/RandomSearch',
    project_name='keras_tuning'
)

tuner.search_space_summary()

tuner.search(X_std, y_std,
             epochs=50,
             validation_split=0.2,
             callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

best_model.fit(X_std, y_std, epochs=1000, validation_split=0.2, callbacks=[early_stopping])


# Predictions for the test set
y_pred = best_model.predict(X_test_std)
y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

# # Transforming to array and saving
y_pred = np.array(y_pred)
# saveSubmission(y_pred, 'HP_tuned_kerasNetwork')