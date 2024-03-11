#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:42:26 2023

@author: armi
"""

from set_figure_defaults import FigureDefaults  # Copyright Antti Vepsalainen
import os
import matplotlib.pyplot as plt
import datetime
import pickle
import seaborn as sn
import pandas as pd
import numpy as np
from numpy.random import SeedSequence
from hper_bo import bo_sim_target, acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
from scipy.special import erf, erfinv

import scipy as sp

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import tqdm
import time

import logging

from functools import partial

#from hper_bo_noiselevel import predict_points_noisy
from hper_repetitions_noiselevel import create_ternary_starting_points

def predict_points_noisy(x, rs = None):
    if rs is not None:
        np.random.seed(rs)
    
    # Predictions.
    xn0 = np.random.normal(x, 10)
    xn1 = np.random.normal(x, 10)
    
    xn = [xn0, xn1]
    
    return xn

def myfun(m, starting_point_candidates):
    
    x = np.array(starting_points[0])[:,0]
    #print(x)
    
    xn = predict_points_noisy(x, m)
    
    print('\nThread ' + str(m) + ': ' + str(xn) + '\n')


###########################

m_total = 3
# Create a list of seeds for repetitions (increase max_reps if you need
# more repetitions than the current max_rep value is).
max_reps = 200
max_init_pts = 1
starting_points = create_ternary_starting_points(
    n_reps=max_reps, n_init=max_init_pts)

###############################################################################

# Number of cpus available to this job.
try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = mp.cpu_count()
'''
# This is a serial version of the code.
for i in range(m_total):
    
    myfun(i, starting_point_candidates = starting_points)
'''    
# This is a parallelized version of the code.
# Create a pool of workers (corresponding to Ncpus)
with mp.Pool(ncpus) as pool:
    
    # 'Partial' shares the same starting point vector between all the methods.
    # Repetition 0 has the same init points for every method. Repetition 1 has
    # init points differing from repetition 0 but again, shared by each method.
    r = process_map(partial(myfun,
                            starting_point_candidates=starting_points),
                    range(m_total), max_workers=ncpus)