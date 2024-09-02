#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:23 2022

@author: armi
"""
#from set_figure_defaults import FigureDefaults  # Copyright Antti Vepsalainen
#import os
import matplotlib.pyplot as plt
#import datetime
import pickle
#import seaborn as sn
#import pandas as pd
import numpy as np
#from numpy.random import SeedSequence
from hper_bo import bo_sim_target
#from hper_util_bo import acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
#from scipy.special import erf, erfinv

#import scipy as sp

#import multiprocessing as mp
#from tqdm.contrib.concurrent import process_map
#import tqdm
#import time

#import GPyOpt
#import GPy

import psutil

#import logging

#from functools import partial

from hper_util_gp import load_GP_model

#from hper_util_repetitions import create_ternary_starting_points
from hper_util_repetitions import cg, build_filenames, set_bo_settings, set_repeat_settings, modify_filename

# Reduces memory leakage with Spyder IDE. Otherwise not necessary.
import matplotlib
matplotlib.interactive(False)

def repeated_tests(m, starting_point_candidates):#, gt_model_targetprop,
                   #gt_model_human):

    print(' ', end='', flush=True)
    
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, '\n')

    
    c_eig = [0.1, 0.25, 0.75]#[0.05, 0.1, 0.25]#, 1]#[0.75]  # Expected information gain.
    # Size of the exclusion zone in percentage points (max. 100)
    c_exclz = [5, 10, 15]#[5, 10, 15]#, 20]#[20]
    # Gradient limit. When the number is higher, the criterion picks less points. 0.05#, 0.07, 0.1, 0.2, 0.5, 0.75
    c_g = list(cg(np.array([0.2, 0.5, 0.6, 0.8, 0.9])))#list(cg(np.array([0.95, 0.9, 0.8])))#list(cg(np.array([0.6])))

    jitters = [0.01, 0.1, 0.5]

    folder = '$WRKDIR/Results/20240902/Test_ho_long/2/' # $WRKDIR/Results/ for the server
    #ground_truth = [0.165, 0.04, 0.79] #[0.17, 0.03, 0.80]  # From C2a paper

    bo_params = {'n_repetitions': 25, # Repetitions of the whole BO process.
                 'n_rounds': 50, # Number of rounds in one BO.
                 'n_init': 3, # Number of initial sampling points.
                 'batch_size': 2, # Number of samples in each round.
                 'materials': ['CsPbI', 'MAPbI', 'FAPbI'], # Materials, i.e., search space variable names
                 'noise_target': 1  # Noise level of the target variable (between [0,1])
                 }
    
    noise_df = 1 # Noise level of the data fusion variable (between [0,1], used only if data fusion is used)

    # Give False if you don't want to save the figures.
    save_figs = False
    # Give False if you don't want to save disk space while saving the data.
    save_disk_space = True
    # Give True if you want to close the figures immediately after they are created.
    close_figs = True
    
    # Give range(bo_params['n_repetitions']) if you want to run all the repeats.
    # Give specific indices if you want to run only some of them (e.g., the
    # run was interrupted before).
    #indices_of_repeats = range(bo_params['n_repetitions'])
    indices_of_repeats = np.arange(10, 15, 1)
    
    data_fusion_property, df_data_coll_method, acquisition_function, c_grad, c_e, jitter, fetch_file_date = set_repeat_settings(
        m, c_g, c_exclz, c_eig, jitters)
    
    if (m > 0):
        
        ###############
        # Typically, one does not need to modify these inputs.
        
        acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params = set_bo_settings(
            bo_params, acquisition_function, jitter, data_fusion_property, 
            df_data_coll_method, noise_df, c_grad, c_e)
        
        # Create result folders and build filenames for result files.
        pickle_filenames, figure_filenames, triangle_folder = build_filenames(
            folder, bo_params, acq_fun_descr, df_data_coll_descr,
            fetch_file_date=fetch_file_date, m=m)
        
        ###############

        all_starting_points = []
        bo_examples = []
        optima = []
        model_optima = []
        X_accum_all = []
        Y_accum_all = []
        data_fusion_params_all = []
        surrogate_model_params_all =  []
        
        # Initialize starting points for each repeat.
        for i in range(bo_params['n_repetitions']):

            all_starting_points.append(
                starting_point_candidates[i][0:bo_params['n_init']])

            message = ('\n\nInit points method ' + str(m) + 
                         ',  repetition ' + str(i) + ':\n' +
                  str(all_starting_points[i]))
            print(message)
            
        for i in indices_of_repeats:

            # Plot the BO for the first two iterations.
            if (i < 2) and (save_figs == True):
                no_plots = False
            else:
                no_plots = True

            if acq_fun_params is None:

                afp = None

            else:

                afp = acq_fun_params.copy()

            if df_data_coll_params is None:

                ddcp = None
                message = 'Start method ' + str(m) + ': No data fusion, repetition ' + str(i)
                
            else:

                ddcp = df_data_coll_params.copy()
                message = 'Start method ' + str(m) + ': ', ddcp, ', repetition ' + str(i)
                

            print(message)
                            
            next_suggestions, optimum, model_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, bo_objects = bo_sim_target(
                targetprop_data_source = gt_model_targetprop,
                human_data_source = gt_model_human,
                materials=bo_params['materials'],
                rounds=bo_params['n_rounds'],
                init_points=all_starting_points[i],
                batch_size=bo_params['batch_size'],
                acquisition_function=acquisition_function,
                acq_fun_params=afp,
                df_data_coll_params=ddcp,
                no_plots=no_plots, results_folder=triangle_folder,
                noise_target = bo_params['noise_target'],
                seed = None, close_figs = close_figs)
            
            # Getting % usage of virtual_memory ( 3rd field)
            print('BO ended. \n')
            print('RAM memory % used:', psutil.virtual_memory()[2])
            
            
            optima.append(optimum)
            model_optima.append(model_optimum)
            X_accum_all.append(X_accum)
            Y_accum_all.append(Y_accum)

            if data_fusion_params is not None:
                data_fusion_params_all.append(data_fusion_params)
            
            surrogate_model_params_all.append(surrogate_model_params)
            
            if ddcp is None:
                
                message = 'End method ' + str(m) + ': No data fusion, repetition ' + str(i)
                
            else:
                
                message = 'End method ' + str(m) + ': ', ddcp, ', repetition ' + str(i)

            print(message)
            
            # Example BO objects saved only from the first two repetitions
            # to save disk space.
            if (save_disk_space is False) or (i < 2):
                
                bo_examples.append([bo_objects])
                
                filename = modify_filename(pickle_filenames[-1], i+1)

                dbfile = open(filename, 'ab')
                pickle.dump(bo_examples, dbfile)
                dbfile.close()
                
                if (save_disk_space is True) and (i == 1):
                    
                    # The variable is not needed anymore and it tends to be large so let's delete it.
                    del bo_examples
                    
            # Save other results after all repetitions have been done but
            # also  times in between if the total number of repetitions
            # is large.
            if (i == (bo_params['n_repetitions']-1)) or (
                    (bo_params['n_repetitions'] > 10) and
                    (np.remainder((i+1),
                                  int(np.floor(bo_params['n_repetitions']/10)))
                     == 0)):

                pickle_variables = ({'optimal_samples': optima,
                                     'model_optima': model_optima}, 
                                    X_accum_all, Y_accum_all,
                                    surrogate_model_params_all, 
                                    data_fusion_params_all) #, results, lengthscales_all,
                                    #variances_all, max_gradients_all]

                # Save the results as an backup
                for j in range(len(pickle_variables)):
                    
                    # Temporary filename for temp run safe-copies.
                    filename = modify_filename(pickle_filenames[j], i+1)

                    dbfile = open(filename, 'ab')
                    pickle.dump(pickle_variables[j], dbfile)
                    dbfile.close()
                    
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            print('Start next repeat...\n')
            
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, '\n')
        print('Clearing variables...\n')
        
        del next_suggestions, optimum, model_optimum, X_rounds, Y_rounds
        del X_accum, Y_accum, surrogate_model_params, data_fusion_params
        del pickle_variables


if __name__ == "__main__":
    
    ###############################################################################
    # MAIN SETTINGS FOR HITL BENCHMARKS
    
    # Paths to the GPy GPRegression models that will be used for fetching source data.
    path_gtmodel_targetprop = './Source_data/gt_model_traget_variable'#'./Source_data/stability_gt_model_GPR' # GPy.models.gp_regression.GPRegression
    #path_gtmodel_targetprop = './Source_data/stability_model_equal_AB'
    path_gtmodel_humanevals = './Source_data/visualquality/human_model_scale0to1'  # GPy.models.gp_regression.GPRegression
    
    # Number of methods to be tested.
    m_total = 120
    # Indices of methods to be tested. Default: range(m_total)
    indices_methods = range(m_total)
    
    # Load the starting points for BO. Every method will
    # share these same init points.
    # shape: (repeat_idx, init_point_idx, composition)
    init_points = np.load('./Source_data/initpts.npy')
    n_init_points = 10
    
    # DISABLED
    # Generate a list of seeds for repetitions (increase max_reps if you need
    # more repetitions than the current max_rep value is). Every method will
    # share these same init points.
    #max_reps = 50
    #max_init_pts = 3
    #init_points = create_ternary_starting_points(
    #    n_reps=max_reps, n_init=max_init_pts)
    
    init_points = np.array(init_points)
    
    
    
    ###############################################################################
    
    plt.figure()
    
    for ip in range(init_points.shape[1]):
        
        plt.scatter(init_points[:, ip, 0], init_points[:, ip, 1], label = 'P' + str(ip))
        
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('The ' + str(init_points.shape[1]) + ' initial points for each repetition of BO')
    plt.legend()
    plt.show()
    
    print('Sanity-check of the dimensions of the initial points. Compositions should sum up to one, e.g. for the first repeat: ', np.sum(init_points[0, :, :], axis = 1))
    
    ## Number of cpus available to this job.
    #try:
    #    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    #except KeyError:
    #    ncpus = mp.cpu_count()
    #
    #print('Number of CPUs used: ', ncpus)
    
    # Load source data models. The implemented code assumes these models do not
    # output scaled values but data in real units.
    global gt_model_targetprop
    gt_model_targetprop = load_GP_model(path_gtmodel_targetprop) # Perovskite stability data (units in [px*min]), 0 px*min is fully stable and high values are instable
    global gt_model_human
    gt_model_human = load_GP_model(path_gtmodel_humanevals) # Human opinion on sample quality data, scale [0,1], where 1 is bad quality and 0 is high quality.
    
    # This is a serial version of the code.
    for i in indices_methods:
        
        for j in range(1):#n_init_points):
            
            repeated_tests(i, starting_point_candidates = init_points)#[[j],:])#,
                       #gt_model_targetprop = gt_model_targetprop,
                       #gt_model_human = gt_model_human)
    '''    
    # This is a parallelized version of the code.
    # Create a pool of workers (corresponding to Ncpus)
    with mp.Pool(ncpus) as pool:
        
        r = process_map(partial(repeated_tests,
                                starting_point_candidates=init_points),
                        indices_methods, max_workers=ncpus)
    
    '''

