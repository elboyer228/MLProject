import numpy as np

import keras
from keras import layers
from keras_tuner import BayesianOptimization
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from tools import selectFeatures, getTarget, saveSubmission

from Models.kerasNetwork import kerasNetwork

# Reproducibility global settings
seed_num = 1
keras.utils.set_random_seed(seed_num)


################ cddd_bayesian_large.csv - kaggle public score: 0.18831 ################

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
    directory="repro",
    project_name="repro"
)


# tuner.search(X_std, y_std,
#              epochs=100,
#              validation_data=(X_val_std, y_val_std),
#              callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])

# tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Retrieve parameters only
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

best_model.fit(X_std, y_std, epochs=1000, batch_size=70 ,validation_data=(X_val_std, y_val_std), callbacks=[EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)])

# Predictions for the test set
y_pred = best_model.predict(X_test_std)
y_pred = Yscaler.inverse_transform(y_pred).reshape(-1)

# # Transforming to array and saving
y_pred = np.array(y_pred)
saveSubmission(y_pred, "export#2")



################ cddd_bayesian_large_repro.csv - kaggle public score: 0.20814 ################

model_structure = [
    layers.Dense(512, activation='relu'),
    layers.Dense(192, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(224, activation='relu'),
    layers.Dense(320, activation='relu'),
    layers.Dense(224, activation='relu'),
    layers.Dense(384, activation='relu'),
    layers.Dense(480, activation='relu'),
    layers.Dense(416, activation='relu'),
    layers.Dense(1)
]

kerasNetwork(
    *selectFeatures(Lab = True, mol = True, cddd=True, bestCddd=100),
    getTarget(),
    model_structure=model_structure,
    seed_num=seed_num,
    batch_size=70, 
    epochs=1000, 
    lr=0.0016,
    decay=0.004,
    patience=150,
    verbose=True,
    export=False,
    name="repro#1")