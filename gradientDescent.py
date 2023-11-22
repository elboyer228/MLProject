from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
from tools import saveSubmission
import matplotlib.pyplot as plt
import numpy as np 



train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
train_properties = pd.read_csv("Features/train_properties.csv")
test_properties = pd.read_csv("Features/test_properties.csv")

# Define features and target
enhanced = ["MolWt", "NumValenceElectrons", "MaxPartialCharge","BCUT2D_LOGPHI", "BCUT2D_MRHI", "BCUT2D_CHGLO", "Kappa2", "PEOE_VSA6" ]
target = "RT"


X = train_properties.loc[:, enhanced]
X_test = test_properties.loc[:, enhanced]
y = train[target]

# Defining model and scaling data
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter = 10000, eta0 = 0.001, tol=1e-3, early_stopping=True))
reg.fit(X, y)

# Predicting and saving data
predicted = reg.predict(X_test)
saveSubmission(predicted, "SGDRegression_" + "enhanced")


predictedTraining = reg.predict(X)

# Computing metrics
mae = mean_absolute_error(y, predictedTraining)
mse = mean_squared_error(y, predictedTraining)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictedTraining)

print(r2)


plt.figure()
plt.scatter(y, predictedTraining, s=2, label="Stoca Gradient Descent Regression")
x = np.linspace(min(y), max(y), 100) 
plt.plot(x, x, color='black', linestyle='--')
plt.xlabel("Experimental RT")
plt.ylabel("Predicted RT")
plt.title("Stoca Gradient Descent Regression")
plt.legend()
plt.text(0.1,0.9,f"R2 Score: {r2_score(y, predictedTraining)}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}",transform=plt.gca().transAxes,verticalalignment='top')
plt.savefig('Visualization_data/GradientDescent.png', format='png')
