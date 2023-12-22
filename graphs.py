import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

from tools import selectFeatures, getTarget

from Models.kerasNetwork import kerasNetwork
from Models.neuralNetworks import neuralNetworks



def bestMolGraph():
    # Graphing effect of bestMol on validation loss
    bestMol = 0
    bestValLoss = 100
    val_losses = []
    mols = list(range(1, 211, 10))

    for i in tqdm(mols):
        model, history, val_loss = kerasNetwork(*selectFeatures(Lab = True, mol = True, cddd=True, bestMol=i, bestCddd=100), getTarget(), batch_size=128, verbose=False)
        val_losses.append(val_loss)
        if val_loss < bestValLoss:
            bestValLoss = val_loss
            bestMol = i
            print(f"Best number of molecular features: {bestMol}, Validation loss: {bestValLoss}")

    print(f"Best number of molecular features: {bestMol}, Validation loss: {bestValLoss}")

    # Plotting the validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(mols, val_losses, marker='o')
    plt.title('Validation loss vs Number of molecular features')
    plt.xlabel('Number of molecular features')
    plt.ylabel('Validation loss')
    plt.grid(True)
    plt.show()


def bestCdddGraph():
    # Graphing effect of bestCddd on validation loss
    bestCddd = 0
    bestValLoss = 100
    val_losses = []
    cddds = list(range(10, 251, 10))

    for i in tqdm(cddds):
        model, history, val_loss = kerasNetwork(*selectFeatures(Lab = True, mol = True, cddd=True, bestCddd=i), getTarget(), name=i, batch_size=128, verbose=False,)
        val_losses.append(val_loss)
        if val_loss < bestValLoss:
            bestValLoss = val_loss
            bestCddd = i
            print(f"Best number of CDDD features: {bestCddd}, Validation loss: {bestValLoss}")

    print(f"Best number of CDDD features: {bestCddd}, Validation loss: {bestValLoss}")

    # Plotting the validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(cddds, val_losses, marker='o')
    plt.title('Validation loss vs Number of CDDD features')
    plt.xlabel('Number of CDDD features')
    plt.ylabel('Validation loss')
    plt.grid(True)
    plt.show()
    
    
# Graph the effect of patience on validation loss
def patienceGraph():
    bestPatience = 0
    bestValLoss = 100
    val_losses = []
    patiences = list(range(10, 501, 50))

    for i in tqdm(patiences):
        model, history, val_loss = kerasNetwork(*selectFeatures(Lab = True, mol = True, cddd=True, bestCddd=100), getTarget(), batch_size=128, patience=i, verbose=False)
        val_losses.append(val_loss)
        if val_loss < bestValLoss:
            bestValLoss = val_loss
            bestPatience = i
            print(f"Best patience: {bestPatience}, Validation loss: {bestValLoss}")

    print(f"Best patience: {bestPatience}, Validation loss: {bestValLoss}")

    # Plotting the validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(patiences, val_losses, marker='o')
    plt.title('Validation loss vs Patience')
    plt.xlabel('Patience')
    plt.ylabel('Validation loss')
    plt.grid(True)
    plt.show()
    
    
# Plot different learning rates
def plotLR():
    X_set, X_test = selectFeatures(Lab=True, mol=True) 
    y_set = getTarget()
    learning_rates = np.linspace(0.00001, 0.001, 10)
    train_losses = []
    validation_losses = []
    for lr in learning_rates:
        train_loss, validation_loss = neuralNetworks(X_set, X_test, y_set, seed_num=2, batch_size=128, epochs=1500, lr=lr, patience=50, verbose=False, export=False, name="NeuralNetwork")
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(f"Learning rate: {lr}, Train loss : {train_losses[-1]}, Validation loss: {validation_losses[-1]}")
        
    plt.plot(learning_rates, validation_losses)
    plt.xlabel("Learning rate")
    plt.ylabel("Validation loss")
    plt.xscale("log")
    plt.show()