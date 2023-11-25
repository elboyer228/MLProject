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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from tools import selectFeatures, getTarget, saveSubmission



def neuralNetworks(X_set, X_test, y_set, seed_num=2, batch_size=128, epochs=150, lr=0.0001, patience=50, verbose=True, export=False, name="NeuralNetwork", k_fold=5):
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


    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
             print(f'Reset trainable parameters of layer = {layer}')
             layer.reset_parameters()


    # Setting up the model
    class NeuralNet(torch.nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(X_set.shape[1], 128),
                torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
        def forward(self, x):
            return self.layers(x)


    model = NeuralNet()
    model.apply(reset_weights)

    best_val_loss = float('inf')
    patience_counter = 0

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    kf = KFold(n_splits= k_fold, shuffle=True, random_state=seed_num)

     
    for fold, (train_index, val_index) in enumerate(kf.split(X_set)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_sampling = SubsetRandomSampler(train_index)
        test_sampling = SubsetRandomSampler(val_index)

        X_train_datas = DataLoader(X_set, batch_size=batch_size, sampler=train_sampling)
        X_val_datas = test_sampling.utils.data.DataLoader(X_set, batch_size=batch_size, sampler=test_sampling)
        # avec les y_train et y_val dedans 


# training part 
        for batch_train in X_train_datas:
            X_train, y_train = batch_train

            Xscaler = StandardScaler()
            X_std = Xscaler.fit_transform(X_train)
            Yscaler = StandardScaler()
            y_std = Yscaler.fit_transform(y_train.values.reshape(-1, 1))
            
            X_ten, y_ten = map(
                lambda array: torch.tensor(array, dtype=torch.float32),
                [X_std, y_std]
            )

            training_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_ten, y_ten),
                batch_size = batch_size,
                shuffle = True,
                worker_init_fn = seed_worker, # for reproducibility
                generator = g # for reproducibility
            )

            model.train()
            for epoch in tqdm(range(epochs)):
                for data, target in training_dataloader:
                    optimizer.zero_grad()
                    pred = model(data)
                    loss = torch.sqrt(loss_func(pred, target))
                    loss.backward()
                    optimizer.step()
            
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs}', f'Loss: {loss.item()}', end='\r')
                
            # Validation
                model.eval()
                with torch.no_grad():
                    pred_val = model(X_val_ten)
                    val_loss = torch.sqrt(loss_func(pred_val, y_val_ten)) 
                    
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping")
                        break

# validation part
        for batch_val in X_val_datas:
            X_val, y_val = batch_val

            Xscaler = StandardScaler()
            X_val_std = Xscaler.transform(X_val)
            Yscaler = StandardScaler()
            y_val_std = Yscaler.transform(y_val.values.reshape(-1, 1))

            X_val_ten, y_val_ten = map(
                lambda array: torch.tensor(array, dtype=torch.float32),
                [X_val_std, y_val_std]
            )

            model.eval()
            with torch.no_grad():
                pred_train = model(X_ten)
                pred_val = model(X_val_ten)

                # Inverse transform the predictions
                pred_train = Yscaler.inverse_transform(pred_train.numpy())
                pred_val = Yscaler.inverse_transform(pred_val.numpy())

                # Convert the targets back to numpy and inverse transform
                y_train_np = Yscaler.inverse_transform(y_ten.numpy())
                y_val_np = Yscaler.inverse_transform(y_val_ten.numpy())

                # Compute the loss with the inverse transformed data
                train_loss = torch.sqrt(loss_func(torch.from_numpy(pred_train), torch.from_numpy(y_train_np)))
                val_loss = torch.sqrt(loss_func(torch.from_numpy(pred_val), torch.from_numpy(y_val_np)))

                print(f"Train loss: {train_loss}")
                print(f"Validation loss: {val_loss}")
                
                # if export:
                #     predictions = Yscaler.inverse_transform(model(X_test_ten).detach().numpy())
                #     saveSubmission(predictions.flatten(), name)
            
            
    return val_loss





X_set, X_test = selectFeatures(Lab=True, mol=True) 
y_set = getTarget()

neuralNetworks(X_set, X_test, y_set, seed_num=2, batch_size=128, epochs=150, lr=0.0001, patience=50, verbose=False, export=False, name="NeuralNetwork")

# # Plot different learning rates
# learning_rates = np.linspace(0.00001, 0.001, 10)
# validation_losses = []
# for lr in learning_rates:
#     validation_losses.append(neuralNetworks(X_set, X_test, y_set, seed_num=2, batch_size=128, epochs=1500, lr=lr, patience=50, verbose=False, export=False, name="NeuralNetwork"))
    
# plt.plot(learning_rates, validation_losses)
# plt.xlabel("Learning rate")
# plt.ylabel("Validation loss")
# plt.xscale("log")
# plt.savefig("Analysis/NeuralNetworks/learning_rates.png")