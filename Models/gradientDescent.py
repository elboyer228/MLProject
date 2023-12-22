# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tools import saveSubmission

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf

from tools import selectFeatures, getTarget, plotHistory, saveSubmission

# Reproducibility
X_set, X_test = selectFeatures(Lab=True, mol=True)
y_set = getTarget()

# Splitting test and train

seed_num = 42
X_train, X_val, y_train, y_val = train_test_split(
    X_set,
    y_set,
    test_size=0.20,
    random_state=seed_num
)


# Defining model and scaling data
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter = 10000, eta0 = 0.001, tol=1e-3, early_stopping=True))
reg.fit(X_train, y_train)

# Predicting and saving data
predicted_X_val = reg.predict(X_val)


# Computing metrics
mae = mean_absolute_error(y_val, predicted_X_val)
mse = mean_squared_error(y_val, predicted_X_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predicted_X_val)

print(f'The r2_score is: {r2}')
print(f'The MSE is: {mse}')
print(f'The RMSE is: {rmse}')

plt.figure()
plt.scatter(y_val, predicted_X_val, s=2, label="Stoca Gradient Descent Regression")
x = np.linspace(min(y_val), max(y_val), 100) 
plt.plot(x, x, color='black', linestyle='--')
plt.xlabel("Experimental RT")
plt.ylabel("Predicted RT")
plt.title("Stoca Gradient Descent Regression")
plt.legend()
plt.text(0.1,0.9,f"R2 Score: {r2_score(y_val, predicted_X_val)}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}",transform=plt.gca().transAxes,verticalalignment='top')
plt.savefig('Visualization/GradientDescentLab.png', format='png')


predicted_X_test = reg.predict(X_test)
saveSubmission(predicted_X_test, "SGD/SGDRegressionLab")
