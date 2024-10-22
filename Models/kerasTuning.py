# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np


import tensorflow as tf
import keras
from keras import layers
from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.losses import mean_squared_error
from keras import regularizers
from keras.layers import BatchNormalization, Activation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools import selectFeatures, getTarget, saveSubmission



# Reproducibility
seed_num = 1
keras.utils.set_random_seed(seed_num)

project_name = 'repro'


X_set, X_test = selectFeatures(Lab=True, mol=True, cddd=True, bestCddd=100)
y_set = getTarget()

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


def build_model_tunned(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=X_set.shape[1]))
    for i in range(hp.Int('num_layers', 5, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=1024,
                                            step=32), activation='relu'))
    model.add(layers.Dense(1))
    
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate = 0.0016,
            weight_decay = 0.004
        ),
        loss="mean_squared_error",
    )
    return model

tuner = BayesianOptimization(
    build_model_tunned,
    objective='val_loss',
    max_trials=50,
    seed = seed_num,
    overwrite=False,
    directory='hptuning/'+project_name,
    project_name=project_name
)

tuner.search_space_summary()

tuner.search(X_std, y_std,
             epochs=100,
             validation_data=(X_val_std, y_val_std),
             callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Retrieve parameters only
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

best_model.fit(X_std, y_std, epochs=1000, batch_size=70 ,validation_data=(X_val_std, y_val_std), callbacks=[EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)])

# Predictions for the validation set
y_pred_val = best_model.predict(X_val_std)
y_pred_val = Yscaler.inverse_transform(y_pred_val).reshape(-1)

y_pred_train = best_model.predict(X_std)
y_pred_train = Yscaler.inverse_transform(y_pred_train).reshape(-1)

# Compute mean squared error
val_loss = mean_squared_error(y_val, y_pred_val)
loss = mean_squared_error(y_train, y_pred_train)


# Inverse transform the losses and print
print(f"Train loss : {loss}, Validation loss: {val_loss}")

# Predictions for the test set
y_pred = best_model.predict(X_test_std)
y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

# # Transforming to array and saving
y_pred = np.array(y_pred)
saveSubmission(y_pred, "export#1")




