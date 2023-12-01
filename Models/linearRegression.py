# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import saveSubmission

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def LinearRegressionModel(type = "normal"):
    
    # Import the data
    train = pd.read_csv("Data/train.csv")
    test = pd.read_csv("Data/test.csv")
    train_properties = pd.read_csv("Features/train_properties.csv")
    test_properties = pd.read_csv("Features/test_properties.csv")
    
    # Define features and target
    normal = ["MolWt", "NumValenceElectrons", "MaxPartialCharge"]
    enhanced = ["MolLogP", "BertzCT", "NumAliphaticHeterocycles"]
    more_features = ["MolWt", "NumValenceElectrons", "MaxPartialCharge","BCUT2D_LOGPHI", "BCUT2D_MRHI", "BCUT2D_CHGLO", "Kappa2", "PEOE_VSA6" ]
    target = "RT" 

    
    if type == "fingerprints":
        X = train.loc[:, 'ECFP_1':'ECFP_1024']
        X_test = test.loc[:, 'ECFP_1':'ECFP_1024']
        y = train[target]
    elif type == "normal":
        X = train_properties.loc[:, normal]
        X_test = test_properties.loc[:, normal]
        y = train[target]
    elif type == "enhanced":
        X = train_properties.loc[:, enhanced]
        X_test = test_properties.loc[:, enhanced]
        y = train[target]
    elif type == "more_features":
        X = train_properties.loc[:, more_features]
        X_test = test_properties.loc[:, more_features]
        y = train[target]

    # Defining model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting and saving data
    predicted = model.predict(X_test)
    saveSubmission(predicted, "LinearRegression/LinearRegression_" + type)


    # Testing the fit by plotting predictions on the training data
    predictedTraining = model.predict(X)
    # Computing metrics
    mae = mean_absolute_error(y, predictedTraining)
    mse = mean_squared_error(y, predictedTraining)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictedTraining)


    plt.figure()
    plt.scatter(y, predictedTraining, s=2, label="Linear Regression")
    x = np.linspace(min(y), max(y), 100) 
    plt.plot(x, x, color='black', linestyle='--') # Plot identity line
    plt.title("Linear Regression (" + type + ")")
    plt.xlabel("true counts")
    plt.ylabel("predicted mode of counts")
    plt.legend()
    plt.text(0.1,0.9,f"R2 Score: {r2_score(y, predictedTraining)}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}",transform=plt.gca().transAxes,verticalalignment='top')
    plt.savefig(
        'Visualization_data/LinearRegression_' + type + '.png', format='png'
    )

LinearRegressionModel("fingerprints")
LinearRegressionModel("normal")
# LinearRegressionModel("enhanced")
# LinearRegressionModel("more_features")