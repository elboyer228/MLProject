# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import selectFeatures, getTarget, saveSubmission
from tqdm import tqdm

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


seed_num = 1


def LinearRegressionModel(Lab = True, mol = True, cddd = True, ECFP = False, bestCddd = 100, export = False, plot = False, name_submission = "lastSubmission"):
    
    X_set, X_test = selectFeatures(Lab=Lab, mol=mol, cddd=cddd, ECFP=ECFP, bestCddd=bestCddd)
    y_set = getTarget()
    
    # Splitting test and train
    
    X_train, X_val, y_train, y_val = train_test_split(
    X_set,
    y_set,
    test_size=0.20,
    random_state=seed_num
)

    # Defining model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Testing the fit by plotting predictions on the validation data
    predicted_val = model.predict(X_val)
    
    # Computing metrics
    mae = mean_absolute_error(y_val, predicted_val)
    mse = mean_squared_error(y_val, predicted_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predicted_val)

    if plot:
        plt.figure()
        plt.scatter(y_val, predicted_val, s=2, label="Linear Regression")
        x = np.linspace(min(y_val), max(y_val), 100) 
        plt.plot(x, x, color='black', linestyle='--') # Plot identity line
        plt.title("Linear Regression - Validation test")
        plt.xlabel("true values")
        plt.ylabel("predicted values")
        plt.legend()
        plt.text(0.1,0.9,f"R2 Score: {r2}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}",transform=plt.gca().transAxes,verticalalignment='top')
        plt.savefig('Visualization_data/LinearRegression_.png', format='png')
        plt.show()
    if export:
        predicted = model.predict(X_test)
        saveSubmission(predicted, "LinearRegression/LinearRegression_" + name_submission + ".csv")
    
    return mse, r2


LinearRegressionModel(Lab=True, mol=True, cddd=True, ECFP=False, bestCddd=100, export=False, plot=True, name_submission='LabMolCdddDataset')