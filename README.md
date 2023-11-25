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

## Authors

* Elise Boyer ([@elboyer228](https://github.com/elboyer228))
* Johann Clausen ([@johanncc01](https://github.com/Johanncc01))

## Files

The given dataset is stored in the `data` folder. It has been enhanced with additional features, and the resulting dataset is stored in the [`full_train_data.csv`](Data/full_train_data.csv) and the [`full_test_data.csv`](Data/full_test_data.csv) files.

To access a set of features from one of these files, the following shortcuts can be used :

```python
Lab_data = train.loc[:, 'Lab_1':'Lab_24']
```

```python
ECFP_fingerprints = train.loc[:, 'ECFP_1':'ECFP_1024']
```

```python
cddd_fingerprints = train.loc[:, 'cddd_1':'cddd_512']
```

```python
molecular_properties = train.loc[:, 'MaxAbsEStateIndex':'fr_urea']
```