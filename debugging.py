#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:49:30 2023

@author: armi
"""

import os
import pickle
import GPyOpt

def load_ground_truth(path_model):
    
    # Load already existing stability data as the "ground truth" of stability.
    with open(path_model,'rb') as f:
        target_model_raw = pickle.load(f)
    
    # The saved model is GPyOpt GPModel that is a wrapper of GPy gp_regression.
    # GPyOpt is not maintained anymore, so for easier adaptation to other
    # packages, we use here GPy gp_regression model. Let's dig it out.
    target_model = target_model_raw.model
    
    
    return target_model



print(os.getcwd())
path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222'
print(os.path.isfile(path))

stability_model = load_ground_truth(path)