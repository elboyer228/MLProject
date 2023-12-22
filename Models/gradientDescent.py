# Including parent folder for imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np 
import matplotlib.pyplot as plt
from tools import saveSubmission

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tools import selectFeatures, getTarget, saveSubmission

def StochGradient(seed_num = 1, Lab = True, mol = True, cddd = True, ECFP = False, bestCddd = 100, export = False, plot = False, name_submission = "SGDLastSubmission"):
    X_set, X_test = selectFeatures(Lab=Lab, mol=mol, cddd=cddd, ECFP=ECFP, bestCddd=bestCddd)
    y_set = getTarget()

    # Splitting test and train
    X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.20, random_state=seed_num)

    # Defining model and scaling data
    sgd = SGDRegressor(max_iter = 10000, eta0 = 0.001, tol=1e-3, early_stopping=True)
    param_grid = {
        'eta0': [0.001, 0.01, 0.1, 1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Create a pipeline
    pipeline = make_pipeline(StandardScaler(), GridSearchCV(sgd, param_grid, cv=5))

   
    pipeline.fit(X_train, y_train)
    best_params = pipeline.named_steps['gridsearchcv'].best_params_

    print(f'Best parameters: {best_params}')

    predicted_X_val = pipeline.predict(X_val)

    # Computing metrics
    mae = mean_absolute_error(y_val, predicted_X_val)
    mse = mean_squared_error(y_val, predicted_X_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predicted_X_val)

    if plot:
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
        plt.show()

    predicted_X_test = pipeline.predict(X_test)

    if export:
        saveSubmission(predicted_X_test, "SGD/SGDRegression_" + name_submission + ".csv")



StochGradient(seed_num = 1, Lab = True, mol = True, cddd = True, ECFP = False, bestCddd = 100, export = True, plot = True, name_submission = "LabMolCddd")