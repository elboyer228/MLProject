# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

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

# Define features and target
X = train.drop(["ID"], axis=1)
X_test = test.drop(["ID"], axis=1)
y = target["RT"]

# Standardize data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_test_std = scaler.transform(X_test)
y_std = scaler.fit_transform(y.values.reshape(-1, 1))

# Convert data to pytorch tensors
X_ten, y_ten, X_test_ten = map(
    lambda array: torch.tensor(array, dtype=torch.float32),
    [X_std, y.values.reshape(-1, 1), X_test_std]
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

model = NeuralNet()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

# Training
model.train()
for epoch in tqdm(range(epochs)):
    for data, target in training_dataloader:
        optimizer.zero_grad()
        pred = model(data)
        loss = torch.sqrt(loss_func(pred, target))
        loss.backward()
        optimizer.step()


# Evaluation
with torch.no_grad():
    model.eval()
    train_preds = model(X_ten).detach().numpy()
    train_gt = y_ten.detach().numpy()
    train_loss = np.sqrt(mean_squared_error(train_preds, train_gt))
    
    predictions = model(X_test_ten).detach().numpy().flatten()
    saveSubmission(predictions, "NeuralNetwork_properties")
