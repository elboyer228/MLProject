import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import saveSubmission

# Sklearn
from sklearn.linear_model import LinearRegression



# Import the data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
train_properties = pd.read_csv("Features/train_properties.csv")
test_properties = pd.read_csv("Features/test_properties.csv")

target = "RT"

# Merging training data and computed features
LightTrain = train[["Compound", "SMILES", "Lab", "RT"]]
LightProp = train_properties[["MolWt", "NumValenceElectrons", "MaxPartialCharge", "HeavyAtomCount", "RingCount", "MolLogP", "MolMR"]]
newTrain = pd.concat([LightTrain, LightProp], axis=1)

LightTest = test[["Compound", "SMILES", "Lab"]]
LightTProp = test_properties[["MolWt", "NumValenceElectrons", "MaxPartialCharge", "HeavyAtomCount", "RingCount", "MolLogP", "MolMR"]]
newTest = pd.concat([LightTest, LightTProp], axis=1)

# Defining model
features = ["MolWt", "NumValenceElectrons", "MaxPartialCharge"]
model = LinearRegression()
model.fit(newTrain[features], newTrain[target])

# Predicting and saving data
predicted = model.predict(newTest[features])
saveSubmission(predicted, "LinearRegression")


# Testing the fit by plotting predictions on the training data
predictedTraining = model.predict(newTrain[features])
plt.figure()
plt.scatter(newTrain[target], predictedTraining, s=2, label="Linear Regression")
x = np.linspace(min(newTrain[target]), max(newTrain[target]), 100) 
plt.plot(x, x, color='black', linestyle='--') # Plot identity line
plt.xlabel("true counts")
plt.ylabel("predicted mode of counts")
plt.legend()
plt.show()