#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:23 2022

@author: armi
"""

from Fun_hper_BO import hper_bo
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import datetime
import matplotlib.pyplot as plt

from set_figure_defaults import FigureDefaults # C Antti Vepsalainen


# Turn to function, repeat 10 times.


def ternary_rand():
    x = np.random.rand()
    y = np.random.rand()*(1-x)
    z = 1 - x - y
    
    return [x, y, z]

def ternary_rand_vector(n):
    
    v = []
    
    for i in range(n):
        v.append(ternary_rand())
        
    return v

for m in range(2):

    n_repetitions = 3 #50
    n_rounds = 50 #100
    acquisition_function = 'EI_DFT'
    beta = 0.5 # dft_beta = 0.025, visual_beta = 0.5
    n_init = 2
    batch_size = 2
    if m == 0:
        data_fusion_property = 'visual'
    if m == 1:
        data_fusion_property = 'dft'
    
    # Give True if you don't want to run new BO but only fetch old results and re-plot them.
    fetch_old_results = False
    # Give False if you don't want to save the figures.
    save_figs = True
    # Which data to fetch (if you only fetch and do not calculate new)?
    fetch_file_date = '202203290320'#'202203252333'
    
    ###############
    # Typically, one does not need to modify these inputs.
    if acquisition_function == 'EI_DFT':
        acquisition_description = 'EI_' + data_fusion_property
    else:
        acquisition_description = acquisition_function
    
    fetch_file_postfix = ('_nreps' + str(n_repetitions) + '_nrounds' + str(n_rounds) +
                  '_acq' + acquisition_description + '_beta' + str(beta) +
                  '_inits' + str(n_init) + '_batch' + str(batch_size))
    
    ground_truth = [0.17, 0.03, 0.80] # From C2a paper
    
    pickle_prefix = './Results/Results-'
    pickle_variable_names = ['optima', 'X_steps', 'Y_steps', 'all']
    pickle_postfix = ('_nreps' + str(n_repetitions) + '_nrounds' + str(n_rounds) +
                  '_acq' + acquisition_description + '_beta' + str(beta) +
                  '_inits' + str(n_init) + '_batch' + str(batch_size))
    
    # Set figure style.
    mystyle = FigureDefaults('nature_comp_mat_sc')
    
    
    ###############
    
    if fetch_old_results == False:
    
        all_starting_points = []
        results = []
        optima = []
        X_steps = []
        Y_steps = []
    
        for i in range(n_repetitions):
            
            all_starting_points.append(ternary_rand_vector(n_init))
            
            optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step = hper_bo(
                n_rounds, all_starting_points[i], batch_size, acquisition_function, beta, no_plots = False, data_fusion_property = data_fusion_property)
            
            #results.append([optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step])
            results.append([BO_batch, x_next])
            
            optima.append(optimum)
            
            X_steps.append(X_step)
            Y_steps.append(Y_step)
            print('Repetition ', i)
        
        filename_prefix = pickle_prefix + '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
        pickle_variables = [optima, X_steps, Y_steps, results]
        
        # Save the results as an backup
        for i in range(len(pickle_variable_names)):
            print('Saving variable ', pickle_variable_names[i])
            filename = filename_prefix + '_' + pickle_variable_names[i] + pickle_postfix
            dbfile = open(filename, 'ab') 
            pickle.dump(pickle_variables[i], dbfile)                      
            dbfile.close()
        
    else:
        
        filename_prefix = pickle_prefix + fetch_file_date
        pickle_variables = []
        
        # Fetch the results from pickled backup
        for i in range(len(pickle_variable_names)):
            filename = filename_prefix + '_' + pickle_variable_names[i] + pickle_postfix
            dbfile = open(filename, 'rb') 
            pickle_variables.append(pickle.load(dbfile))                      
            dbfile.close()
            
        optima = pickle_variables[0]
        X_steps = pickle_variables[1]
        Y_steps = pickle_variables[2]
        results = pickle_variables[3]
        
        
    optima = np.array(optima)/60
    
    #mean = np.mean(optima)
    
    
    cols = ['Optimum'+ x for x in list(map(str, range(n_rounds)))]
    df_optima_wide = pd.DataFrame(optima, columns = cols)#range(n_rounds))
    df_optima_wide['Repeat'] = range(n_repetitions) 
    df_optima_long = pd.wide_to_long(df_optima_wide, stubnames = 'Optimum', i = 'Repeat', j = 'Round')
    
    plt.figure()
    sn.lineplot(data = df_optima_long, x = 'Round', y = 'Optimum', ci = 90)
    plt.ylim([0,450000/60])
    plt.tight_layout()
    
    if save_figs:
        
        plt.gcf().savefig(filename + '_optimum.pdf', transparent = True)
        plt.gcf().savefig(filename + '_optimum.svg', transparent = True)
        plt.gcf().savefig(filename + '_optimum.png', dpi=300)
    
    plt.show()
    
    # X regret
    
    regrets = [[None for i in range(n_rounds)] for i in range(n_repetitions)]
    
    for i in range(n_repetitions):
    
        Y_step_all = Y_steps[i]#results[i][9]
        X_step_all = X_steps[i]#results[i][10]
    
        for j in range(n_rounds):
    
            idx_optimum = np.argmin(Y_step_all[j])
            X_optimum = X_step_all[j][idx_optimum]
            regret = np.sqrt(np.sum((ground_truth - X_optimum)**2))
            regrets[i][j] = regret
    
        #print(Y_step_all[j][-1])
        print(regret, X_optimum)
    
    cols = ['Regret'+ x for x in list(map(str, range(n_rounds)))]
    df_regrets_wide = pd.DataFrame(regrets, columns = cols)#range(n_rounds))
    df_regrets_wide['Repeat'] = range(n_repetitions) 
    df_regrets_long = pd.wide_to_long(df_regrets_wide, stubnames = 'Regret', i = 'Repeat', j = 'Round')
    
    plt.figure()
    sn.lineplot(data = df_regrets_long, x = 'Round', y = 'Regret', ci = 90)
    plt.ylim([0,1])
    plt.tight_layout()
    
    if save_figs:
    
        plt.gcf().savefig(filename + '_regret.pdf', transparent = True)
        plt.gcf().savefig(filename + '_regret.svg', transparent = True)
        plt.gcf().savefig(filename + '_regret.png', dpi=300)
    
    plt.show()
