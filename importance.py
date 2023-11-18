from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np


df = pd.read_csv('Features/train_properties.csv')
df_target = pd.read_csv('Data/train.csv')


X = df.drop(columns=['ID'])
y = df_target['RT']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Importance of all characteristics
importances = model.feature_importances_
features = X.columns

# Create a df and saves it
feature_importances = {features[i]: importances[i] for i in range(len(features))}
df_importances = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
df_importances = df_importances.sort_values('Importance', ascending=False)
df_importances.to_csv("Features/importance.csv")


# Also computes the permutation importance and saves it
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

perm_importances = {features[i]: perm_importance.importances_mean[i] for i in range(len(features))}
df_perm_importances = pd.DataFrame(list(perm_importances.items()), columns=['Feature', 'Permutation Importance'])
df_perm_importances = df_perm_importances.sort_values('Permutation Importance', ascending=False)
df_perm_importances.to_csv("Features/permutation_importance.csv")