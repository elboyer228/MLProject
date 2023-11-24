# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch

from tools import saveSubmission



# Reproducibility
seed_num = 2
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



# Load data
train = pd.read_csv("Features/train_properties.csv")
test = pd.read_csv("Features/test_properties.csv")
target = pd.read_csv("Data/train.csv")

# Splitting test and train

X_train, X_val, y_train, y_val = train_test_split(
    train.drop(["ID"], axis=1),
    target["RT"],
    test_size=0.2,
    random_state=42
)

# Standardize data
Xscaler = StandardScaler()
X_std = Xscaler.fit_transform(X_train)
X_val_std = Xscaler.transform(X_val)
X_test_std = Xscaler.transform(test.drop(["ID"], axis=1))
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
    batch_size = 128,
    shuffle = True,
    worker_init_fn = seed_worker, # for reproducibility
    generator = g # for reproducibility
)

# Setting up the model
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(210, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.layers(x)

batch_size = 128
epochs = 150

# Initialize best validation loss as infinity
best_val_loss = float('inf')
# Initialize patience counter
patience_counter = 0

model = NeuralNet()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

# Training
model.train()
for epoch in tqdm(range(epochs)):
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
        pred_val = model(X_val_ten)  # assuming X_val_ten is your validation data
        val_loss = torch.sqrt(loss_func(pred_val, y_val_ten))  # assuming y_val_ten is your validation target

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:  # 5 is the patience
            print("Early stopping")
            break


# Evaluation and computation of the train and test losss
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
    test_loss = torch.sqrt(loss_func(torch.from_numpy(pred_val), torch.from_numpy(y_val_np)))

    print(f"Train loss: {train_loss}")
    print(f"Validation loss: {test_loss}")
    
    predictions = Yscaler.inverse_transform(model(X_test_ten).detach().numpy())
    saveSubmission(predictions.flatten(), "NeuralNetwork_early_stopping")
