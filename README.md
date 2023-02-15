
# HPER
Utilizing human in Bayesian optimization loop, case perovskite stability optimization
=======

HPER 
===========

## Description

Utilizing human in Bayesian optimization loop, case perovskite stability optimization. This repository contains the code base for producing the tests presented as a poster contribution in AI4Mat workshop as a part of Neurips 2022:

[1] Tiihonen, A., Filstroff, L., Mikkola, P., Lehto, E., Kaski, S., Todorović, M., & Rinke, P. More trustworthy Bayesian optimization of materials properties by adding human into the loop. In AI for Accelerated Materials Design NeurIPS 2022 Workshop. https://openreview.net/forum?id=JQSzcd_Zc62

This repository is fitted only for running simulations and needs modifications for running live Bayesian optimization loops.

This is work in progress - please follow the updates in the repository (or drop an email) for more information.

## Description of human in the loop approach

Data fusion approach presented in Github repository SPProC has been further generalized here. The implementation can take in arbitrary data fusion data as csv files. If new data fusion data arrives within each BO round, their corresponding csv files are fed in in a vector format.

The data fusion approach has additional hyperparameters: a scaling factor for scaling data fusion data into a probability distribution (beta), gradient limit for determining when to query humans (g), and the radius of an exclusion zone determining how far the point needs to be from the points previously evaluated by the humans before it can be queried on humans (r). Additional parameters include the hyperparameters for the GP model that is fitted on the human evaluation data.

Most of the code works for any dimensionality of the search space (not yet tested).

## Description of data

The data included in this repository is originally derived in our previous article:

[2] Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter 4(4), 1305-1322, 2021. https://doi.org/10.1016/j.matt.2021.01.008.

The data shared inside the "Source_data" folder is sufficient to reproduce the simulations and figures described in article [1].
- "Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222": Perovskite stability data is in the simulations sampled from a GPRegression (GPy package) model originally produced as a part of article [2] (instead of really conducting perovskite aging tests).
- "Source_data/visualquality/visualquality_round_0-1.csv": Human opinion on sample quality (grades given based on photos taken of the samples after 0min of degradation as a part of article [2]) is provided as integer grades 0-3, then scaled to range [0,1]. These human opinions are used in the code for fitting a "ground truth" GPRegression model (that is sampled in the simulations instead of really asking humans).

## Installation
To install, clone the following repository and sub-repository:

`$ git clone https://github.com/TadaSanar/HPER.git`

`$ cd HPER`

`$ cd GPyOpt_DFT`

`$ git clone https://github.com/TadaSanar/GPyOpt_DFT`

To install the modified GPyOpt package, create a virtual environment using Anaconda (optional but recommended setup):

`$ conda create --name hper python=3.7`

`$ conda activate hper`

Run the following terminal commands to setup the package:

`$ python setup.py install`

`$ pip install -r requirements.txt`

Run HPER/hper_repetitions.py

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen | 
| **VERSION**      | 0.1 / November, 2021 /  derived from https://github.com/PV-Lab/SPProC and https://github.com/PV-Lab/GPyOpt_DFT |
| | 0.2 / November, 2022 /  The latest version, presented at AL4Mat workshop in Neurips 2022 conference |

## Citation

@Misc{hper2022,
  author =   {The HPER authors},
  title =    {{HPER}: Utilizing human in Bayesian optimization loop, case perovskite stability optimization},
  howpublished = {\url{https://github.com/TadaSanar/HPER}},
  year = {2022}
}

@inproceedings{
hper2022ai4mat,
title={More trustworthy Bayesian optimization of materials properties by adding human into the loop},
author={Armi Tiihonen and Louis Filstroff and Petrus Mikkola and Emma Lehto and Samuel Kaski and Milica Todorovi{\'c} and Patrick Rinke},
booktitle={AI for Accelerated Materials Design NeurIPS 2022 Workshop},
year={2022},
url={https://openreview.net/forum?id=JQSzcd_Zc62}
}

Earlier work on the data fusion property:

@Misc{spproc2020,
  author =   {The SPProC authors},
  title =    {{SPProC}: Sequential learning with Physical Probabilistic Constraints},
  howpublished = {\url{https://github.com/PV-Lab/SPProC}},
  year = {2020}
}

Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter, 2021, https://doi.org/10.1016/j.matt.2021.01.008.