import numpy as np

from keras import layers

from tools import selectFeatures, getTarget, saveSubmission

from Models.kerasNetwork import kerasNetwork


################ kerasNetwork_eliseontop.csv - kaggle public score: 0.19697 ################

model_elise = [
            layers.Dense(416, activation='relu'),
            layers.Dense(352, activation='relu'),
            layers.Dense(288, activation='relu'),
            layers.Dense(416, activation='relu'),
            layers.Dense(1)
        ]


repro1 = kerasNetwork(
    *selectFeatures(Lab = True, mol = True, cddd=True, bestCddd=100),
    getTarget(),
    model_structure=model_elise,
    seed_num=1,
    batch_size=70,
    epochs=1000,
    lr=0.0016,
    decay=0.004,
    patience=150,
    verbose=True,
    export=False,
    name="Reproducibility/#1"
    )[-1]

saveSubmission(repro1, "Reproducibility/kerasNetwork_eliseontop")

################ cddd_bayesian_large_repro.csv - kaggle public score: 0.20814 ################

model_bayesian = [
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

repro2 = kerasNetwork(
    *selectFeatures(Lab = True, mol = True, cddd=True, bestCddd=100),
    getTarget(),
    model_structure=model_bayesian,
    seed_num=1,
    batch_size=70, 
    epochs=1000, 
    lr=0.0016,
    decay=0.004,
    patience=150,
    verbose=True,
    export=False,
    name="repro2"
    )[-1]

saveSubmission(repro2, "Reproducibility/cddd_bayesian_large_repro")