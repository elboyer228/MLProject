import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from tools import molPropToCSV

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


# Remove correlated predictors
print("-------------- Correlated predictors --------------")

# Compute and remove correlated predictors from both train and test
# Training set
train_before = clean_train_properties.shape[1]
corr_matrix = clean_train_properties.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] == 1)]
clean_train_properties = clean_train_properties.drop(to_drop, axis=1)
train_after = clean_train_properties.shape[1]

print("Train properties predictors went from", train_before, "to", train_after, "after removing correlated predictors.")

# Test set
test_before = clean_test_properties.shape[1]
clean_test_properties = clean_test_properties.drop(to_drop, axis=1) # We use the same to_drop as for the train set
test_after = clean_test_properties.shape[1]

print("Test properties predictors went from", test_before, "to", test_after, "after removing correlated predictors.")



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
train_data = pd.concat([train_data, train.loc[:, 'ECFP_1':'ECFP_1024']], axis=1)
test_data = pd.concat([test_data, test.loc[:, 'ECFP_1':'ECFP_1024']], axis=1)

# Adding cddd fingerprints
train_data = pd.merge(train_data, cddd, on='SMILES', how='left')
test_data = pd.merge(test_data, cddd, on='SMILES', how='left')

# Adding properties
train_data = pd.concat([train_data, train_properties], axis=1)
test_data = pd.concat([test_data, test_properties], axis=1)

# Saving complete data
train_data.to_csv("Data/full_train_data.csv", index=False)
test_data.to_csv("Data/full_test_data.csv", index=False)

