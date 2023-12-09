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

# Reproducibility
seed_num = 42


def LinearRegressionModel(num = 0, type = "mol", plot = False, export = False):
    
    if type == "mol":
        X_set, X_test = selectFeatures(Lab=True, mol=True, bestFeatures=num)
    elif type == "cddd":
        X_set, X_test = selectFeatures(Lab=True, cddd=True)
    elif type == "ECFP":
        X_set, X_test = selectFeatures(Lab=True, ECFP=True)
    elif type == "both":
        X_set, X_test = selectFeatures(cddd=True, ECFP=True)

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
        plt.title("Linear Regression (" + type + ") - Validation test")
        plt.xlabel("true values")
        plt.ylabel("predicted values")
        plt.legend()
        plt.text(0.1,0.9,f"R2 Score: {r2}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}",transform=plt.gca().transAxes,verticalalignment='top')
        plt.savefig('Visualization_data/LinearRegression_' + type + '.png', format='png')
        plt.show()

    if export:
        predicted = model.predict(X_test)
        saveSubmission(predicted, "LinearRegression/LinearRegression_" + type)
    
    return mse, r2

def plotFeaturesEv():
    plt.figure(figsize=(10,6))
    mse_list = []
    r2_list = []

    for i in tqdm(range(210)):
        mse, r2 = LinearRegressionModel(num=i)
        mse_list.append(mse)
        r2_list.append(r2)
        
    plt.plot(mse_list, label="mse (lower is better)")
    plt.plot(r2_list, label="r2 (higher is better)")
    plt.xlabel("nb of features")
    plt.ylabel("error")
    plt.title("Evolution of the error depending on the feature number (linear reg)")
    plt.legend()
    plt.savefig("Visualization_data/LinearRegression_comparaison.png")

LinearRegressionModel(num=0, type="mol", plot=True, export=True)