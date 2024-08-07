
# HPER
Utilizing human in Bayesian optimization loop, case perovskite stability optimization
=======

HPER 
===========

## Description

Utilizing human in Bayesian optimization loop, case perovskite stability optimization. This repository is fitted only for running simulations and needs (limited) modifications for running live Bayesian optimization loops.

This is work in progress - please follow the updates in the repository (or drop an email) for more 
information.

## Branches

Branch 'Neurips2022' contains the code base for producing the tests presented as a poster contribution in AI4Mat workshop as a part of Neurips 2022:

[1] Tiihonen, A., Filstroff, L., Mikkola, P., Lehto, E., Kaski, S., Todorović, M., & Rinke, P. More trustworthy Bayesian optimization of materials properties by adding human into the loop. In AI for Accelerated Materials Design NeurIPS 2022 Workshop. https://openreview.net/forum?id=JQSzcd_Zc62

Branch 'feat-eig' is stable and uses Python 3.7.

Branch 'main' is under development and uses Python 3.12.


## Description of human in the loop approach

Data fusion approach presented in Github repository SPProC has been further generalized here. The implementation can take in arbitrary data fusion data as csv files. If new data fusion data arrives within each BO round, their corresponding csv files are fed in in a vector format.

The data fusion approach has additional hyperparameters: two scaling factors for scaling data fusion data into a probability distribution (p_beta, p_midpoint), a gradient limit for determining when to query humans (g), and - depending on which option the user has chosen to use - either the radius of an exclusion zone or the expected information gain determining how far the point needs to be from the points previously evaluated by the humans before it can be queried on humans (e).

Further additional parameters include the hyperparameters for the GP model that is fitted on the human evaluation data.

Most of the code works for any dimensionality of the search space (not yet tested).

## Description of data

The data included in this repository is originally derived in our previous article:

[2] Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter 4(4), 1305-1322, 2021. https://doi.org/10.1016/j.matt.2021.01.008.

The data shared inside the "Source_data" folder is sufficient to reproduce the simulations and figures described in article [1].

- "Source_data/stability_gt_model_GPR": Perovskite stability data is in the simulations sampled from a GPRegression model (GPy package) originally produced as a part of article [2] (instead of really conducting perovskite aging tests).
- "Source_data/visualquality/human_model_scale0to1": Human opinion on sample quality (grades given based on photos taken of the samples after 0min of degradation as a part of article [2]) is provided as integer grades 0-3, then scaled to range [0,1]. These human opinions are used in the code for fitting a "ground truth" GPRegression model (that is sampled in the simulations instead of really asking humans).

## Installation

To install, clone the following repository and sub-repository:

`$ git clone https://github.com/TadaSanar/HPER.git .`
`$ git clone https://github.com/TadaSanar/GPyOpt.git ./GPyOpt_DF`
`$ git checkout main`
`$ cd GPyOpt_DF`
`$ git checkout gpyopt-df`
`$ conda env create -f environment_temp_across_platforms_versions_fixed.yml`
`$ conda activate hper2`

Run HPER/hper_repetitions_less_memory.py

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen | 
| **VERSION**      | 0.1 / November, 2021 /  Python 3.7., derived from https://github.com/PV-Lab/SPProC and https://github.com/PV-Lab/GPyOpt_DFT |
| | 0.2 / November, 2022 /  Presented at AL4Mat workshop in Neurips 2022 conference |
| | 0.3 / August, 2024 /  The latest version, Python 3.12 |

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

@article{sun2021data,
  title={A data fusion approach to optimize compositional stability of halide perovskites},
  author={Sun, Shijing and Tiihonen, Armi and Oviedo, Felipe and Liu, Zhe and Thapa, Janak and Zhao, Yicheng and Hartono, Noor Titan P and Goyal, Anuj and Heumueller, Thomas and Batali, Clio and others},
  journal={Matter},
  volume={4},
  number={4},
  pages={1305--1322},
  year={2021},
  publisher={Elsevier}
}
