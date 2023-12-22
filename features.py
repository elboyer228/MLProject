import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


from cgi import test
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from tools import selectFeatures, getTarget, saveSubmission



# Import data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
cddd = pd.read_csv("Data/cddd.csv")

#################### Feature computing ####################

# Computing and adding properties for train and test sets
# molPropToCSV(train, "train")
# molPropToCSV(test, "test")

train_properties = pd.read_csv('Features/train_properties.csv').drop(["ID"], axis=1)
test_properties = pd.read_csv('Features/test_properties.csv').drop(["ID"], axis=1)



#################### Data cleaning ####################

# Find if there is missing data
print("-------------- Missing Data --------------")
print("Train properties : ", train_properties.isnull().sum().sum())
print("Test properties : ", test_properties.isnull().sum().sum())
print("Train ECFP : ", train.isnull().sum().sum())
print("Test ECFP : ", test.isnull().sum().sum())
print("CDDD : ", cddd.isnull().sum().sum())

# Replace missing data in cddd properties using SimpleImputer (either "mean" or "constant", which fills with None)
choice = "mean"
# choice = "constant"
cddd_smiles = cddd['SMILES']
cddd_without_smiles = cddd.drop('SMILES', axis=1)
imp = SimpleImputer(missing_values=np.nan, strategy=choice)
imp.fit(cddd_without_smiles)
cddd_without_smiles = pd.DataFrame(imp.transform(cddd_without_smiles), columns=cddd_without_smiles.columns)
clean_cddd = pd.concat([cddd_smiles, cddd_without_smiles], axis=1)

# Find if there are constant predictors and remove them
print("-------------- Constant predictors --------------")
print("Train properties : ", (train_properties.nunique() == 1).sum())
print("Test properties : ",(test_properties.nunique() == 1).sum())
print("Train ECFP : ", (train.nunique() == 1).sum())
print("Test ECFP : ", (test.nunique() == 1).sum())
print("CDDD : ", (cddd.nunique() == 1).sum())

# Remove constant predictors from test properties to both
ConstantProperties = test_properties.columns[test_properties.nunique() == 1]
clean_train_properties = train_properties.drop(ConstantProperties.values, axis=1)
clean_test_properties = test_properties.drop(ConstantProperties.values, axis=1)

# Remove constant predictors from test ECFP to both
ConstantECFP = test.columns[test.nunique() == 1]
clean_train = train.drop(ConstantECFP.values, axis=1)
clean_test = test.drop(ConstantECFP.values, axis=1)



#################### Feature engineering ####################

# Encoding Lab provenance and adding it to the data
enc = OneHotEncoder(handle_unknown='ignore')
enc_train = enc.fit_transform(train["Lab"].values.reshape(-1,1)).toarray()
enc_test = enc.transform(test["Lab"].values.reshape(-1,1)).toarray()

column_names = ['Lab_' + str(i) for i in range(1, 25)]
Lab_train = pd.DataFrame(enc_train, columns=column_names)
Lab_test = pd.DataFrame(enc_test, columns=column_names)

print("-------------- Lab provenance encoding --------------")

print("The lab provenance list is alpahbetical, so the encoding is the same for both train and test sets. \nIt can be found under Features/lab_provenance_encoding.csv.")

# Saving lab provenance encoding
flat_categories = [item for sublist in enc.categories_ for item in sublist]
lab_list = pd.DataFrame({'Lab': column_names, 'Category': flat_categories})
lab_list.to_csv('Features/lab_provenance_encoding.csv', index=False)



#################### Data merging ####################

# Merging data from all sources
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Adding basic info
train_data = pd.concat([train_data, train.loc[:, ["Compound", "SMILES", "RT"]]], axis=1)
test_data = pd.concat([test_data, test.loc[:, ["Compound", "SMILES"]]], axis=1)

# Adding lab provenance encoding
train_data = pd.concat([train_data, Lab_train], axis=1)
test_data = pd.concat([test_data, Lab_test], axis=1)

# Adding ECFP fingerprints
train_data = pd.concat([train_data, clean_train.loc[:, 'ECFP_1':'ECFP_1024']], axis=1)
test_data = pd.concat([test_data, clean_test.loc[:, 'ECFP_1':'ECFP_1024']], axis=1)

# Adding cddd fingerprints
train_data = pd.merge(train_data, clean_cddd, on='SMILES', how='left')
test_data = pd.merge(test_data, clean_cddd, on='SMILES', how='left')

# Adding properties (after experimental tryouts, we decided to keep all properties, not only the ones that are not constant)
train_data = pd.concat([train_data, train_properties], axis=1)
test_data = pd.concat([test_data, test_properties], axis=1)

# Saving complete data
train_data.to_csv("Data/full_train_data.csv", index=False)
test_data.to_csv("Data/full_test_data.csv", index=False)








############ New features selection ############

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

    important_test_data = X_test[selected_feature_names]
    important_test_data = important_test_data.rename(columns={important_test_data.columns[0]: 'first_selected_feature'})

    if Features == 'cddd': 
        cols_to_drop = [f'cddd_{i}' for i in range(1, 513)]
    
        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv(f'Data/Selected/select_features_full_train_set_{number_of_important_features}.csv', index=False)
    
        
        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_test_data], axis=1)

        select_features_full_test_set.to_csv(f'Data/Selected/select_features_full_test_set_{number_of_important_features}.csv', index=False)
    

    if Features == 'ECFP':
        
        cols_to_drop = [f'ECFP_{i}' for i in range(1, 1025)]
        
        
        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv(f'Data/Selected/select_features_full_train_set_{number_of_important_features}.csv', index=False)

        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_test_data], axis=1)
        select_features_full_test_set.to_csv(f'Data/Selected/select_features_full_test_set_{number_of_important_features}.csv', index=False)
        
    if Features == 'both':

        cols_to_drop_train = full_train_set.columns[27:1558]
        cols_to_drop_test = full_test_set.columns[26:1557]

        select_features_full_train_set = full_train_set.drop(columns=cols_to_drop_train)
        select_features_full_train_set = pd.concat([select_features_full_train_set, important_data], axis=1)
        select_features_full_train_set.to_csv(f'Data/Selected/select_features_full_train_set_{number_of_important_features}.csv', index=False)

        select_features_full_test_set = full_test_set.drop(columns=cols_to_drop_test)
        select_features_full_test_set = pd.concat([select_features_full_test_set, important_test_data], axis=1)
        select_features_full_test_set.to_csv(f'Data/Selected/select_features_full_test_set_{number_of_important_features}.csv', index=False)



findMostImportantFeatures(Features='cddd', number_of_important_features=250)