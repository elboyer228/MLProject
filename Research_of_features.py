import pandas as pd
from cgi import test
from tools import molPropToCSV



# Feature computing for train and test sets
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

molPropToCSV(train, "train")
molPropToCSV(test, "test")

train_prop = pd.read_csv('Features/train_properties.csv')

if 'ID' in train_prop.columns:
    train_prop.drop(columns=['ID'], inplace=True)

features_analysis = train_prop.describe().drop('count')
features_analysis.to_csv('Features/Features_analysis.csv')


features_analysis_clean = features_analysis.copy()


for column in features_analysis_clean.columns:
    if 0 <= features_analysis_clean[column]['mean'] <= 0.02:
        features_analysis_clean.drop(columns=[column], inplace=True)


features_analysis_clean.to_csv('Features/Features_analysis_clean.csv')

print(f'number of columns before cleaning: {len(features_analysis.columns)}')
print( f'number of columns after cleaning: {len(features_analysis_clean.columns)}')