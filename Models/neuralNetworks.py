# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch

from tools import selectFeatures, getTarget, plotHistory, saveSubmission



def neuralNetworks(X_set, X_test, y_set, seed_num=2, batch_size=128, epochs=150, lr=0.0001, patience=50, plotHistory = False, verbose=True, export=False, name="NeuralNetwork"):
    """
    This function trains a basic neural network model on the given dataset.

    Parameters
    ----------
    X_set : pandas.DataFrame or numpy.ndarray
        The training dataset.
    X_test : pandas.DataFrame or numpy.ndarray
        The test dataset.
    y_set : pandas.Series or numpy.ndarray
        The target variable for the training dataset.
    seed_num : int, optional
        The seed for the random number generator, by default 2.
    batch_size : int, optional
        The batch size for training the model, by default 128.
    epochs : int, optional
        The number of epochs to train the model, by default 150.
    lr : float, optional
        The learning rate for the optimizer, by default 0.0001.
    patience : int, optional
        The patience for early stopping, by default 50.
    plotHistory : bool, optional
        If True, plots the training and validation loss, by default False.
    verbose : bool, optional
        If True, prints out the loss for each epoch, by default True.
    export : bool, optional
        If True, saves the model after training, by default False.
    name : str, optional
        The name of the model, by default "NeuralNetwork".

    Returns
    -------
    None
        This function doesn't return anything but prints out the training and validation loss.
    """
    # Reproducibility
    torch.use_deterministic_algorithms(True)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed_num)


    X_train, X_val, y_train, y_val = train_test_split(X_set,y_set,test_size=0.20,random_state=seed_num)

    # Standardize data
    Xscaler = StandardScaler()
    X_std = Xscaler.fit_transform(X_train)
    X_val_std = Xscaler.transform(X_val)
    X_test_std = Xscaler.transform(X_test)
    Yscaler = StandardScaler()
    y_std = Yscaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_std = Yscaler.transform(y_val.values.reshape(-1, 1))

    # Convert data to pytorch tensors
    X_ten, y_ten, X_val_ten, y_val_ten, X_test_ten = map(
        lambda array: torch.tensor(array, dtype=torch.float32),
        [X_std, y_std, X_val_std, y_val_std, X_test_std]
    )

    # Pytorch training data loader, for using batches
    training_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_ten, y_ten),
        batch_size = batch_size,
        shuffle = True,
        worker_init_fn = seed_worker, 
        generator = g 
    )

    # Setting up the model
    class NeuralNet(torch.nn.Module):
        def __init__(self, input_size):
            super(NeuralNet, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, 416),  # Use input_size here
                torch.nn.ReLU(),
                torch.nn.Linear(416, 352),
                torch.nn.ReLU(),
                torch.nn.Linear(352, 288),
                torch.nn.ReLU(),
                torch.nn.Linear(288, 416),
                torch.nn.ReLU(),
                torch.nn.Linear(416, 1),
            )
        def forward(self, x):
            return self.layers(x)
        

    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    model = NeuralNet(X_set.shape[1])
    loss_func = torch.nn.HuberLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    model.train()
    iterator = tqdm(range(epochs)) if not verbose else range(epochs)
    for epoch in iterator:
        model.train()
        for data, target in training_dataloader:
            optimizer.zero_grad()
            pred = model(data)
            loss = torch.sqrt(loss_func(pred, target))
            loss.backward()
            optimizer.step()
            
            
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_ten)
            val_loss = (loss_func(pred_val, y_val_ten)).numpy()
            
        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} ({round((epoch/epochs)*100, 1)}%)', f'Training Loss: {loss.item()}', f"Validation loss : {val_loss}", end='\r')
        
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
            
        # Early stopping if the validation loss doesn't improve for a certain number of epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if not verbose:
                    iterator.close()
                print("\n ----- Early stopping -----")
                break


    # Evaluation
    model.eval()
    with torch.no_grad():
        
        pred_train = model(X_ten).detach()
        pred_val = model(X_val_ten).detach()

        # Inverse transform the predictions
        pred_train = Yscaler.inverse_transform(pred_train.numpy())
        pred_val = Yscaler.inverse_transform(pred_val.numpy())
        y_train = Yscaler.inverse_transform(y_ten.numpy())
        y_val = Yscaler.inverse_transform(y_val_ten.numpy())

        # Compute the loss with the inverse transformed data
        train_loss = loss_func(torch.from_numpy(pred_train), torch.from_numpy(y_train))
        val_loss = loss_func(torch.from_numpy(pred_val), torch.from_numpy(y_val))

        r2 = r2_score(y_val, pred_val)
        print(f"R^2 score for validation set: {r2}")
        # Prediction on test set
        if export:
            predictions = Yscaler.inverse_transform(model(X_test_ten).detach().numpy())
            saveSubmission(predictions.flatten(), name)
        
        if plotHistory: 
            plotHistory(history)

    return train_loss, val_loss



# X_set, X_test = selectFeatures(Lab=True, mol=True, cddd=True, ECFP=False, bestCddd=100) 
# y_set = getTarget()

# train_loss, val_loss = neuralNetworks(X_set, X_test, y_set, seed_num=1, batch_size=128, epochs=1500, lr=0.00023, patience=150, plotHistory = False, verbose=False, export=False, name="NeuralNetwork_LabMol")
# print(f"Train loss : {train_loss}, Validation loss: {val_loss}")

# These settings gave the best results: R^2 score for validation set: 0.9868045981517677
# Train loss : 0.0007032371941022575, Validation loss: 0.08734024316072464
# Kaggle score: 0.1913