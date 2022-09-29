
# HPER
Utilizing human in Bayesian optimization loop, case perovskite stability optimization
=======

HPER 
===========
## Description

Utilizing human in Bayesian optimization loop, case perovskite stability optimization.

The current repository is fitted only for running simulations and needs modifications for running live Bayesian optimization loops.

Data fusion approach presented in repository SPProC has been further generalized here. The implementation can take in arbitrary data fusion data as DataFrames (later on in the live Bayesian optimization loop the input will be csv files). If new data fusion data arrives within each BO round, their corresponding csv files are fed in in a vector format.

The data fusion approach has additional hyperparameters: a scaling factor for scaling data fusion data into a probability distribution (beta), gradient limit for determining when to query humans (g), and the radius of an exclusion zone determining how far the point needs to be from the points previously evaluated by the humans before it can be queried on humans (r). Additional parameters include the hyperparameters for the GP model that is fitted on the human evaluation data.

Most of the code works for any dimensionality of the search space (not yet tested).

This is work in progress - please follow the updates in the repository (or drop an email) if you want to know when the work is in a stable mode.

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

Article details coming!

@Misc{hper2022,
  author =   {The HPER authors},
  title =    {{HPER}: Utilizing human in Bayesian optimization loop, case perovskite stability optimization},
  howpublished = {\url{https://github.com/TadaSanar/HPER}},
  year = {2022}
}

Earlier work on the data fusion property:

@Misc{spproc2020,
  author =   {The SPProC authors},
  title =    {{SPProC}: Sequential learning with Physical Probabilistic Constraints},
  howpublished = {\url{https://github.com/PV-Lab/SPProC}},
  year = {2020}
}

Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter, 2021, https://doi.org/10.1016/j.matt.2021.01.008.