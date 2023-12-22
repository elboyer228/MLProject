import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

if not os.path.exists("Submissions/"):
    os.makedirs("Submissions/")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from tools import selectFeatures, getTarget, saveSubmission
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def LassoRegression(seed_num = 1, Lab = True, mol = True, cddd = True, ECFP = False, bestCddd = 100, export = False, plot = False, name_submission = "lastSubmission"):
    
    X_set, X_test = selectFeatures(Lab=Lab, mol=mol, cddd=cddd, ECFP=ECFP, bestCddd=bestCddd)
    y_set = getTarget()
    
    # Splitting test and train
    
    X_train, X_val, y_train, y_val = train_test_split(X_set,y_set, test_size=0.15,random_state=seed_num)
    
    model = Lasso(max_iter=10000)
    param_grid = {'alpha':  np.linspace(1, 10,40)}

    
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=0)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print(f'Best parameters: {best_params}')
    model = Lasso(max_iter=1000,**best_params)
    model.fit(X_train, y_train)

    predicted_val = model.predict(X_val)

    
    mae = mean_absolute_error(y_val, predicted_val)
    mse = mean_squared_error(y_val, predicted_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predicted_val)

    if plot:
        plt.figure()
        plt.scatter(y_val, predicted_val, s=2, label="Lasso Regression")
        x = np.linspace(min(y_val), max(y_val), 100) 
        plt.plot(x, x, color='black', linestyle='--')
        plt.title("Lasso Regression - Validation test")
        plt.xlabel("true values")
        plt.ylabel("predicted values")
        plt.legend()
        plt.text(0.1,0.9,f"R2 Score: {r2}\nRMSE: {rmse}\nMSE: {mse}\nMAE: {mae}, \nalpha: {best_params['alpha']}",transform=plt.gca().transAxes,verticalalignment='top')
        plt.savefig('Visualization_data/LassoRegression.png', format='png')
        plt.show()

    if export:
        predicted = model.predict(X_test)
        saveSubmission(predicted, "LassoRegression/LassoRegression_" + name_submission + ".csv")


LassoRegression(seed_num = 1, Lab = True, mol = True, cddd = True, ECFP = False, bestCddd = 100, export = True, plot = True, name_submission = "LabMolCddd")
