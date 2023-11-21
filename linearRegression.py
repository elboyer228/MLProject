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

    # Defining model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting and saving data
    predicted = model.predict(X_test)
    saveSubmission(predicted, "LinearRegression_" + type)


    # Testing the fit by plotting predictions on the training data
    predictedTraining = model.predict(X)
    plt.figure()
    plt.scatter(y, predictedTraining, s=2, label="Linear Regression")
    x = np.linspace(min(y), max(y), 100) 
    plt.plot(x, x, color='black', linestyle='--') # Plot identity line
    plt.xlabel("true counts")
    plt.ylabel("predicted mode of counts")
    plt.legend()
    plt.show()

    # Computing metrics
    mae = mean_absolute_error(y, predictedTraining)
    mse = mean_squared_error(y, predictedTraining)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictedTraining)

    print(f"Linear Regression ({type})")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
LinearRegressionModel("fingerprints")
LinearRegressionModel("normal")
LinearRegressionModel("enhanced")