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
model = keras.Sequential([
    keras.Input(shape=X_set.shape[1]),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])


model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanSquaredError()]
)

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