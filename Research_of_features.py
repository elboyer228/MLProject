import pandas as pd
from cgi import test
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from tools import selectFeatures, getTarget, saveSubmission

seed_num = 30
def clean_features(file_path = 'Features/train_properties.csv'):
    train_prop = pd.read_csv(file_path)
    
    if 'ID' in train_prop.columns:
        train_prop.drop(columns=['ID'], inplace=True)
    
    features_analysis = train_prop.describe().drop('count')
    features_analysis.to_csv('Features/Features_analysis.csv')
    
    features_analysis_clean = features_analysis.copy()
    
    for column in features_analysis_clean.columns:
        if 0 <= features_analysis_clean[column]['mean'] <= 0.02:
            features_analysis_clean.drop(columns=[column], inplace=True)
    
    features_analysis_clean.to_csv('Features/Features_analysis_clean.csv')
    
    print(f'Number of columns before cleaning: {len(features_analysis.columns)}')
    print(f'Number of columns after cleaning: {len(features_analysis_clean.columns)}')


# Features impotance analysis of cdd and ECFP features
def findMostImportantFeatures(Features = 'cddd', number_of_important_features=10):
    """
    This function performs feature selection on a given dataset and saves the most important features to a CSV file.

    Parameters:
    ----------
    Features (str): The type of features to consider. Options are 'cddd', 'ECFP', and 'both'.
    number_of_important_features (int): The number of important features to select.

    Returns:
    -------
    None. The function saves the most important features to a CSV file.

    """
    full_train_set = pd.read_csv('Data/full_train_data.csv')
    full_test_set = pd.read_csv('Data/full_test_data.csv')

    if Features == 'cddd':
        X_set, X_test = selectFeatures(cddd=True)
    if Features == 'ECFP':
        X_set, X_test = selectFeatures(ECFP=True)
    if Features == 'both':
        X_set, X_test = selectFeatures(cddd=True, ECFP=True)

    y_set = getTarget()

    X_train, X_val, y_train, y_val = train_test_split(X_set,y_set,test_size=0.20,random_state=seed_num)

    regressor = LinearRegression()

    sfs = SequentialFeatureSelector(regressor, k_features=number_of_important_features, forward=True, scoring='neg_mean_squared_error', cv=5, verbose=1)
    sfs.fit(X_train, y_train)

    selected_features = sfs.k_feature_idx_

    X_train_selected = sfs.transform(X_train)
    X_val_selected = sfs.transform(X_val)

    # Training the model with the selected features
    regressor.fit(X_train_selected, y_train)
    mse = regressor.score(X_val_selected, y_val)
    print("Mean Squared Error: ", mse)

    selected_feature_names = X_train.columns[list(selected_features)]
    important_data = X_set[selected_feature_names]
    important_data = important_data.rename(columns={important_data.columns[0]: 'first_selected_feature'})

    if Features == 'cddd': 
        cols_to_drop = [f'cddd_{i}' for i in range(1, 513)]
    
        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv('Data/select_features_full_train_set.csv', index=False)
        
        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_data], axis=1)

        select_features_full_test_set.to_csv('Data/select_features_full_test_set.csv', index=False)

    if Features == 'ECFP':
        
        cols_to_drop = [f'ECFP_{i}' for i in range(1, 1025)]
        
        
        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv('Data/select_features_full_train_set.csv', index=False)

        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_data], axis=1)
        select_features_full_test_set.to_csv('Data/select_features_full_test_set.csv', index=False)
        
    if Features == 'both':

        cols_to_drop_train = full_train_set.columns[27:1558]
        cols_to_drop_test = full_test_set.columns[26:1557]

        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop_train)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv('Data/select_features_full_train_set.csv', index=False)

        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop_test)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_data], axis=1)
        select_features_full_test_set.to_csv('Data/select_features_full_test_set.csv', index=False)



findMostImportantFeatures(Features='cddd', number_of_important_features=100)