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
    """This function calculates molecular properties for each molecule in the given DataFrame and saves them to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the molecules. Each molecule is represented by its SMILES string.
    name : str
        The name of the output CSV file.
    """
    prop = pd.DataFrame([Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(m)) for m in df["SMILES"]])
    prop.insert(0, "ID", range(1, len(prop)+1))
    prop.to_csv(f"Features/{name}_properties.csv", index=False)


def saveSubmission(predicted, name = "output"):
    """This function saves the predicted RT values to a CSV file in the correct format for submission.

    Parameters
    ----------
    predicted : array-like
        The predicted RT values to be saved.
    name : str, optional
        The name of the output CSV file, by default "output".
    """
    submission = pd.DataFrame({'ID': range(1, len(predicted)+1) , 'RT': predicted})
    submission.to_csv("Submissions/"+name+".csv", index=False)

# Feature computing for train and test sets
molPropToCSV(train, "train")
molPropToCSV(test, "test")