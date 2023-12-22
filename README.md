# MLProject : Predicting retention times in liquid chromatography


> Introduction to Machine Learning for bioengineers - BIO-322



Liquid chromatography is a widely used method for detecting drugs in human tissues and biofluids. In this project, we aim to predict the retention time (RT) of drugs on different chromatography platforms, based on their chemical structure.

In a liquid chromatography experiment, each drug is identified by its unique retention time. This parameter is influenced by the chemical properties of the drug as well as the specific configuration of the chromatography setup within a particular laboratory.

To predict the retention time of a new drug, we have a dataset consisting of molecular structures represented in the SMILES format, along with their corresponding retention times measured in various laboratories using different chromatography platforms.

By leveraging machine learning techniques, we aim to develop accurate and reliable models that can predict drug retention times, which in real life applications would enable faster and more efficient drug analysis in various laboratory settings.

## Contents

- [Authors](#authors)
- [Requirements](#requirements)
- [Installation](#installation)
- [Files](#files)
- [Execution](#execution)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Authors

* Elise Boyer ([@elboyer228](https://github.com/elboyer228))
* Johann Clausen ([@johanncc01](https://github.com/Johanncc01))

## Requirements
The required packages for this project are listed in the [`requirements`](MLProject.yml) file. The main ones are: 
- python = 3.9.18
- numpy
- pandas
- pytorch
- keras


## Installation

To install the project, clone the repository and install the package using the following commands :
    
```bash
git clone https://github.com/elboyer228/MLProject.git
cd MLProject
conda env create -f MLProject.yml
```
This will create a conda environment named `MLProject` with all the required packages.



## Files
- [`Data`](Data): folder containing the training and tests sets used for the project
- [`Features`](Features): folder containing the features analysis files, such as importance and correlation
- [`Models`](Models): folder containing all models functions used for the project
    
    The main models are:
    - jkfdsj
    - klfdsj
    - lkjfsdl


- [`Submissions`](Submissions): folder containing the submissions files for the kaggle competition
- [`Viusalization`](Visualization): folder containing the visualization files

---
- [`data_visualization.py`](data_visualization.py): file containing the functions used for the data visualization, used for analysing RT distribution and correlation between features/labs
- [`graphs.py`](graphs.py): file containing the functions used for the visualization of the models results, such as the learning curves or parameter influence
- [`importance.py`](importance.py): file containing the functions used for the feature importance analysis, such as the permutation importance
- [`MLProject.yml`](MLProject.yml): conda environment file
- [`reproducibility.py`](reproducibility.py): file that must be run to check reproducibility of the results
- [`research_of_features.py`](research_of_features.py): file containing the functions used for the features analysis, such as the correlation between features and the selection of the best features
- [`tools.py`](tools.py): file containing the helper function, such as the data loader `selectFeatures()` and the saving function `saveSubmission()`. These helped to select quickly the features (between ECFP, cddd, etc.) and save the submission files to the correct format for the kaggle competition.

## Execution
To check the reproducibility, activate the conda environment and run the [`reproducibility.py`](reproducibility.py) file:

```bash
conda activate MLProject
python reproducibility.py
```
The results, saved to the [`Submissions/Reproducibility`](Submissions/Reproducibility) folder, will contain the same predictions as the ones submitted to the kaggle competition.

---

Any other function can be run in the same way, for example to visualize the data:

```bash
conda activate MLProject
python data_visualization.py
```

To test a specific model, use the provided functions in the [`Models`](Models) folder.
## Results
The performance of the predictions is evaluated using the root mean squared error (RMSE) between the predicted and the true retention times. The lower the RMSE, the better the model. Results are presented in the table below:

| Model | RMSE (public leaderbord score) |
| --- | --- |
| Stochastic Gradient Descent | 1.10842 |
| Linear | 1.08627 |
| Ridge regression | 1.07444 |
| Neural Network - Pytorch | 0.1913 |
| Neural Networl - Keras | 0.18831 |

Further details on the results can be found on the Kaggle competition page: https://www.kaggle.com/competitions/epfl-bio322-2023/leaderboard

## Acknowledgements
We would like to thank our professor and teaching assistants for their guidance and support throughout this project. 