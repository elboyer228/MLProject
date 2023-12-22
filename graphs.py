import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

from tools import selectFeatures, getTarget

from Models.kerasNetwork import kerasNetwork
from Models.neuralNetworks import neuralNetworks

from scipy.stats import norm

########## Graphing training dataset ##########
def dataGraphs():
    X_train = pd.read_csv('Data/train.csv')

    # number of repeted molecules in the data set 
    X_train['Compound'] = X_train['Compound'].str.strip()
    name_counts = X_train['Compound'].value_counts()
    repeated_compounds = name_counts[name_counts > 1]
    repeated_compounds_df = repeated_compounds.reset_index()
    repeated_compounds_df.to_csv('Visualization/Repeated_Compounds_table.csv', index=False)

    #number of data from labs
    X_train['Lab'] = X_train['Lab'].str.strip()
    lab_counts = X_train['Lab'].value_counts()
    plt.figure(figsize=(14, 8))
    plt.bar(lab_counts.index, lab_counts.values, color='lightblue')
    plt.xticks(rotation=90)
    plt.xlabel('Lab')
    plt.ylabel('Counts')
    plt.title('Counts of Molecules per Lab')
    plt.tight_layout()
    plt.savefig('Visualization/Lab_counts.png')


    # analysis of the Retention time 
    plt.figure(figsize=(10, 6))
    retention_times = X_train['RT']

    mean = np.mean(retention_times)
    std_dev = np.std(retention_times)
    z_scores = [(x - mean) / std_dev for x in retention_times] #to get the standardized 

    plt.hist(retention_times, bins=10, density=True, alpha=0.6, color='lightblue', label='Retention Times')
    plt.hist(z_scores, bins=10, density=True, alpha=0.6, color='lightgreen', label='Standardized Values')

    # Generating the normal distribution curve
    x = np.linspace(min(retention_times), max(retention_times), 100)
    y = norm.pdf(x, mean, std_dev)  
    plt.plot(x, y, 'b--', label='Normal Distribution')

    #Standirdized nomal distribution curve 
    x_std = np.linspace(min(z_scores), max(z_scores), 100)
    y_std = norm.pdf(x_std, 0, 1) 
    plt.plot(x_std, y_std, 'g--', label='Standard Normal Distribution')

    plt.text(mean, 0.20, f'Mean={mean:.2f}', color='blue', fontsize=9, ha='left')
    plt.text(mean, 0.17, f'Std Dev={std_dev:.2f}', color='blue', fontsize=9, ha='left')

                        
    plt.xlabel('Retention Time')
    plt.ylabel('Probability')
    plt.title('Distribution of standardized and non-standardized retention times')
    plt.legend()
    plt.savefig('Visualization/RT_analysis.png')



########## Graphing results from predictions ##########


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