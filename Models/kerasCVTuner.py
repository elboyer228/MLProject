# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np

import keras
import keras_tuner
from keras import layers

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.losses import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

from tools import selectFeatures, getTarget, saveSubmission



# Reproducibility
seed_num = 1
keras.utils.set_random_seed(seed_num)

project_name = 'cddd_bayesian_cv2'

X_set, X_test = selectFeatures(Lab=True, mol=True, cddd=True, bestCddd=100)
y_set = getTarget()


def build_tunned_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=X_set.shape[1]))
    # if hp.Choice('use_bn_', values=[0, 1]):
    #         model.add(layers.BatchNormalization())
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=1024,
                                            step=32),
                               activation='relu'))
            
        # Add Dropout
        # model.add(layers.Dropout(rate=hp.Float('dropout_rate_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
        
        # Add l1 and l2 regularization
        # l1_value = hp.Choice('l1_' + str(i), values=[0.0, 0.01, 0.001, 0.0001])
        # l2_value = hp.Choice('l2_' + str(i), values=[0.0, 0.01, 0.001, 0.0001])
        # if l1_value > 0 or l2_value > 0:
        #     model.add(layers.ActivityRegularization(l1=l1_value, l2=l2_value))
            
    model.add(layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate = hp.Float('learning_rate', min_value=0.001, max_value=0.01, step=0.0005),
            # weight_decay = hp.Float('weight_decay', min_value=0.001, max_value=0.01, step=0.005)
        ),
        loss="mean_squared_error",
    )
    return model


class CVTuner(keras_tuner.engine.tuner.Tuner):
    def __init__(self, *args, seed_num=None, **kwargs):
        super(CVTuner, self).__init__(*args, **kwargs)
        self.seed_num = seed_num
        np.random.seed(self.seed_num)
        keras.utils.set_random_seed(seed_num)
        self.best_val_loss = np.inf
        self.true_val_loss = None
        self.best_true_val_loss = None
        
    def run_trial(self, trial, x, y, batch_size=32, epochs=1, cv = 3, callbacks = None, verbose = 1):
        cv = model_selection.KFold(cv)
        val_losses = []
        true_val_losses = []
        for train_indices, test_indices in cv.split(x):
            # Standardize data
            Xscaler = StandardScaler()
            x_train = Xscaler.fit_transform(x.iloc[train_indices])
            x_test = Xscaler.transform(x.iloc[test_indices])

            # Split the training data into training and validation sets
            x_train, x_val, y_train, y_val = train_test_split(
            x_train, y[train_indices], test_size=0.2, random_state=self.seed_num)

            # Normalize target data
            Yscaler = StandardScaler()
            y_train = Yscaler.fit_transform(y_train.values.reshape(-1, 1))
            y_val = Yscaler.transform(y_val.values.reshape(-1, 1))
            y_test = Yscaler.transform(y[test_indices].values.reshape(-1, 1))
            
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                      callbacks= callbacks, verbose=verbose, validation_data=(x_val, y_val))
            val_losses.append(model.evaluate(x_test, y_test, verbose=0))
            
            # Compute true validation loss
            y_pred = model.predict(x_test, verbose = 0)
            true_val_loss = np.mean((Yscaler.inverse_transform(y_test) - Yscaler.inverse_transform(y_pred))**2)
            true_val_losses.append(true_val_loss)
            
        mean_val_loss, mean_true_val_loss = np.mean(val_losses), np.mean(true_val_losses)
        self.oracle.update_trial(trial.trial_id, {'val_loss': mean_val_loss, 'true_val_loss': mean_true_val_loss})
        self.true_val_loss = mean_true_val_loss
        if mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss
            self.best_true_val_loss = mean_true_val_loss
        self.save_model(trial.trial_id, model)
        
    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        print("\nTrue_val_loss:", self.true_val_loss)
        print("Best True_val_loss so far:", self.best_true_val_loss)
    
    def save_model(self, trial_id, model, step=0):
        fname = self._get_checkpoint_fname(trial_id)
        model.save_weights(fname)
        
        
tuner = CVTuner(
    hypermodel=build_tunned_model,
    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
        objective = keras_tuner.Objective("true_val_loss", direction="min"),
        max_trials = 150),
    seed_num=seed_num,
    overwrite=False,
    directory='hptuning/'+project_name,
    project_name=project_name
)


tuner.search(X_set, y_set, epochs=100, batch_size=70, cv = 3, callbacks=[
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=0.0001, verbose=1)
    ]
)

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Standardize the whole training set
Xscaler = StandardScaler()
X_std = Xscaler.fit_transform(X_set)

# Normalize target data
Yscaler = StandardScaler()
y_std = Yscaler.fit_transform(y_set.values.reshape(-1, 1))

# Split the training data into training and validation sets
X_train_std, X_val_std, y_train_std, y_val_std = train_test_split(
    X_std, y_std, test_size=0.2, random_state=seed_num)

# Fit the best model on the training set
best_model.fit(X_train_std, y_train_std, epochs=1000, batch_size=70, validation_data=(X_val_std, y_val_std), callbacks=[
    EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001, verbose=1)
    ])

# Standardize the test set with the same StandardScaler used for the training set
X_test_std = Xscaler.transform(X_test)

# Predictions for the test set
y_pred = best_model.predict(X_test_std)

# Inverse transform the predictions
y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

# Transforming to array and saving
y_pred = np.array(y_pred)
saveSubmission(y_pred, 'KerasNetworks/'+project_name)