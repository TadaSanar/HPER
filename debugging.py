#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:49:30 2023

@author: armi
"""

import os

print(os.getcwd())
path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222'
print(os.path.isfile(path))