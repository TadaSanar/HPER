<<<<<<< HEAD
# HPER
Utilizing human in Bayesian optimization loop, case perovskite stability optimization
=======

HPER 
===========
## Description

Utilizing human in Bayesian optimization loop, case perovskite stability optimization. Data fusion approach presented in repository SPProC has been further generalized here. It can take in arbitrary data fusion data as csv files, and if new data fusion data arrives within each BO round, their corresponding csv files are fed in in a vector format. The pre-defined scaling factors for both the GPR model and the probability model can be feeded in, too. The constraint can also be defined as an arbitrary function (ongoing work). Most of the code works for any dimensionality of the search space (not yet tested).

## Installation
To install, just clone the following repository and sub-repository:

`$ git clone https://github.com/TadaSanar/HPER.git`

`$ cd HPER`

`$ cd GPyOpt_DFT`

`$ git clone https://github.com/TadaSanar/GPyOpt_DFT`

To install the modified GPyOpt package, create a virtual environment using Anaconda (Optional but recommended setup):

`$ conda create --name hper python=3.7`

`$ conda activate hper`

Run the following terminal commands to setup the package:

`$ python setup.py install`

`$ pip install -r requirements.txt`

Run HPER/Repetitions.py

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen | 
| **VERSION**      | derived from https://github.com/PV-Lab/SPProC and https://github.com/PV-Lab/GPyOpt_DFT / November, 2021     | 
| **EMAILS**      | armi.tiihonen@gmail.com  | 
||                    |

## Attribution
This work is under MIT License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation

    
>>>>>>> 24cd03644e66963cec042031a896af6b79ef27db
