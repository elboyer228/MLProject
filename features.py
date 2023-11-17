import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors



# Import the data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")


def molPropToCSV(df, name):
    prop = pd.DataFrame([Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(m)) for m in df["SMILES"]])
    prop.insert(0, "ID", range(1, len(prop)+1))
    prop.to_csv(f"Features/{name}_properties.csv", index=False)

# Feature computing for train and test sets
molPropToCSV(train, "train")
molPropToCSV(test, "test")