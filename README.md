# MLProject

## Overview

Liquid chromatography is a method to detect drugs in human tissues and biofluids, for example, the cocaine concentration in urine. In a liquid chromatography experiment, each drug can be identified by its so-called retention time (RT). The retention time depends critically on the chemical properties of the drug and the exact configuration of the chromatography within a particular laboratory. For a new drug, the retention time can be measured experimentally by obtaining a clean sample of the new drug and directly measuring its retention time on the chromatography setup in a given laboratory. Alternatively, since the RT depends on the chemical structure, one could measure the retention time on one machine, determine the molecular structure of the new drug and predict the retention time on other machines based on the chemical features of the new drug. This is what we will do in this project. You are given the molecular structure of many drugs in the SMILES format together with the retention times measured in different laboratories, on different chromatography platforms. You can either use the SMILES notation directly or domain-specific features, like circular fingerprints or CDDD embeddings, to fit machine learning models that predict the retention times.


By Élise Boyer & Johann Clausen 
