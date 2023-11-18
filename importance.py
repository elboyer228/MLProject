from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


df = pd.read_csv('Features/train_properties.csv')
df_target = pd.read_csv('Data/train.csv')


X = df.drop(columns=['ID'])
y = df_target['RT']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Importance of all characteristics
importances = model.feature_importances_
features = X.columns

# Create a df and saves it
feature_importances = {features[i]: importances[i] for i in range(len(features))}
df_importances = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
df_importances = df_importances.sort_values('Importance', ascending=False)
df_importances.to_csv("Features/importance.csv")
