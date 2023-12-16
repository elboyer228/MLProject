import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors



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
    
    
def selectFeatures(Lab = False, ECFP = False, cddd = False, bestCddd = 0,  mol = False, bestMol = 0):
    """
    This function selects the specified features from the training and test datasets.
    The features are extracted from the enhanced dataset contained in the Data folder.
    If no features are specified, empty DataFrame are returned.

    Parameters
    ----------
    Lab : bool, optional
        If True, includes features 'Lab_1' to 'Lab_24'. By default, False.
    ECFP : bool, optional
        If True, includes features 'ECFP_1' to 'ECFP_1024'. By default, False.
    cddd : bool, optional
        If True, includes features 'cddd_1' to 'cddd_512'. By default, False.
        If `bestCddd` is specified and is between 1 and 500, selects the `bestCddd` most important cddd features.
    bestCddd : int, optional
        The number of most important cddd features to select. Only applicable when `cddd` is True.
        If set to 0, all cddd features are selected. By default, 0.
    mol : bool, optional
        If True, includes molecular features. If `bestMol` is specified, selects the `bestMol` most important molecular features.
        By default, False.
    bestMol : int, optional
        The number of most important molecular features to select. Only applicable when `mol` is True.
        If set to 0, all molecular features are selected. By default, 0.

    Returns
    -------
    tuple of pandas.DataFrame
        The first DataFrame contains the selected features from the training dataset.
        The second DataFrame contains the selected features from the test dataset.
    """
    
    # If no features are specified, return empty DataFrames
    if not Lab and not ECFP and not cddd and not mol:
        return pd.DataFrame(), pd.DataFrame()

    train = pd.read_csv("Data/full_train_data.csv")
    test = pd.read_csv("Data/full_test_data.csv")
    
    train_features = []
    test_features = []
        
    if Lab:
        Lab_train = train.loc[:, 'Lab_1':'Lab_24']
        train_features.append(Lab_train)
        Lab_test = test.loc[:, 'Lab_1':'Lab_24']
        test_features.append(Lab_test)
    if ECFP:
        ECFP_train = train.loc[:, 'ECFP_1':'ECFP_1024']
        train_features.append(ECFP_train)
        ECFP_test = test.loc[:, 'ECFP_1':'ECFP_1024']
        test_features.append(ECFP_test)
    if mol:
        # if number is between 1 and 210, we select the most important features
        if bestMol > 0 and bestMol <= 210:
            imp = pd.read_csv("Features/permutation_importance.csv")
            
            Lab_train = train[imp.loc[:bestMol-1, 'Feature']]
            train_features.append(Lab_train)
            Lab_test = test[imp.loc[:bestMol-1, 'Feature']]
            test_features.append(Lab_test)
        else:
            molecular_train = train.loc[:, 'MaxAbsEStateIndex':'fr_urea']
            train_features.append(molecular_train)
            molecular_test = test.loc[:, 'MaxAbsEStateIndex':'fr_urea']
            test_features.append(molecular_test)
    if cddd:
        # if number is between 1 and 500, we select the most important features
        if bestCddd > 0 and bestCddd <= 500:
            if bestCddd == 100:
                best_train = pd.read_csv("Data/select_features_full_train_set.csv")
                best_test = pd.read_csv("Data/select_features_full_test_set.csv")
                train_features.append(best_train.loc[:, 'first_selected_feature':])
                test_features.append(best_test.loc[:, 'first_selected_feature':])
            elif bestCddd == 250:
                best_train = pd.read_csv("Data/select_features_full_train_set_250.csv")
                best_test = pd.read_csv("Data/select_features_full_test_set_250.csv")
                train_features.append(best_train.loc[:, 'first_selected_feature':])
                test_features.append(best_test.loc[:, 'first_selected_feature':])
            else:
                cddd_imp = pd.read_csv("Features/cddd_importance.csv")
                train_features.append(train.loc[:, cddd_imp.loc[:bestCddd-1, 'Feature']])
                test_features.append(test.loc[:, cddd_imp.loc[:bestCddd-1, 'Feature']])

    
    return pd.concat(train_features, axis=1), pd.concat(test_features, axis=1)


def getTarget():
    """This function returns the target variable from the training dataset.

    Returns
    -------
    pandas.Series
        The target variable.
    """
    train = pd.read_csv("Data/full_train_data.csv")
    return train.loc[:, 'RT']


def plotHistory(history, type = "PyTorch"):
    """
    This function plots the training and validation loss for a given model's history.

    Parameters
    ----------
    history : dict or keras.callbacks.History
        The history object returned by the model's fit method. 
        For PyTorch, this should be a dictionary with 'train_loss' and 'val_loss' keys. 
        For Keras, this should be a keras.callbacks.History object.

    type : str, optional
        The type of model that generated the history. This should be either 'PyTorch' or 'Keras'. The default is 'PyTorch'.

    Returns
    -------
    None
        This function doesn't return anything; it shows a matplotlib plot.

    Raises
    ------
    ValueError
        If the type parameter is not 'PyTorch' or 'Keras'.
    """
    if type == "PyTorch":
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(history['train_loss'], label='Training loss')
        ax.plot(history['val_loss'], label='Validation loss')
        ax.set_title('Training and validation loss (standardized data)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()
    elif type == "Keras":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.plot(history.history['loss'], label='Training loss')
        ax1.plot(history.history['val_loss'], label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(history.history['accuracy'], label='Training accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.show()
    else:
        raise ValueError("Invalid type. Must be 'PyTorch' or 'Keras'.")