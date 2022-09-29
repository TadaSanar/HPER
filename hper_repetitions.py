#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:23 2022

@author: armi
"""

#from Fun_hper_BO import hper_bo
from hper_bo_simulated_with_df_function import bo_sim_target, ei_dft_param_builder, ei_dft_param2str
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import datetime
import matplotlib.pyplot as plt
import os

from set_figure_defaults import FigureDefaults # Copyright Antti Vepsalainen

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

for m in [1,0,2,3,4]:

    n_repetitions = 25
    n_rounds = 100
    
    n_init = 2
    batch_size = 1
    
    # Give True if you don't want to run new BO but only fetch old results and re-plot them.
    fetch_old_results = False
    # Give False if you don't want to save the figures.
    save_figs = True
    
    if m == 2:
        data_fusion_property = 'visual'
        data_fusion_method = 'model_all'
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = '202208071359'
        
    if m == 4:
        data_fusion_property = 'dft'
        data_fusion_method = 'model_all'
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = '202208071830'
        
    if m == 0:
        data_fusion_property = None
        acquisition_function = 'EI'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = '202208072133'
    
    if m == 1:
        data_fusion_property = 'visual'
        data_fusion_method = 'model_necessary'
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = '202209240248'#'202208080143'
        
    if m == 3:
        data_fusion_property = 'dft'
        data_fusion_method = 'model_necessary'
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = '202208080556'
        
    color = sn.color_palette()[m]
    
    materials = ['CsPbI', 'MAPbI', 'FAPbI']
    
    ###############
    # Typically, one does not need to modify these inputs.
    if acquisition_function == 'EI_DFT':
          acq_fun_params = [ei_dft_param_builder('EI_DFT',
              data_fusion_target_variable = data_fusion_property), 
              data_fusion_property, data_fusion_method]
          acq_fun_param_str = ei_dft_param2str(acq_fun_params)
          acquisition_description = acquisition_function + acq_fun_param_str + '_' + data_fusion_method

    else:
        acquisition_description = acquisition_function
        acq_fun_params = [None, None, None]
        
    fetch_file_postfix = ('_nreps' + str(n_repetitions) + '_nrounds' + str(n_rounds) +
                  '_inits' + str(n_init) + '_batch' + str(batch_size) +
                  '_acq' + acquisition_description)
    
    ground_truth = [0.17, 0.03, 0.80] # From C2a paper
    
    pickle_prefix = './Results/' # Old version: './Results/Results-'
    pickle_variable_names = ['optima', 'X_steps', 'Y_steps', 'data_fusion_data', 'BOmainresults', 'BO_lengthscales', 'BO_variances', 'BO_max_gradients']
    pickle_postfix = fetch_file_postfix
    
    # Set figure style.
    mystyle = FigureDefaults('nature_comp_mat_sc')
    
    
    ###############
    
    if fetch_old_results == False:
    
        all_starting_points = []
        results = []
        optima = []
        X_steps = []
        Y_steps = []
        data_fusion_data_all = []
        lengthscales_all = []
        variances_all = []
        max_gradients_all = []
        
        for i in range(n_repetitions):
            
            
            all_starting_points.append(ternary_rand_vector(n_init))
            
            print(i, materials, n_rounds, all_starting_points[i], batch_size, acquisition_function, acq_fun_params)
            
            # Plot the BO for the first two iterations.
            if i <2:
                no_plots = False
            else:
                no_plots = True
            
            optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step, data_fusion_data, lengthscales, variances, max_gradients = bo_sim_target(
                bo_ground_truth_model_path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222', 
                              materials = materials, rounds = n_rounds,
                              init_points = all_starting_points[i],
                              batch_size = batch_size,
                              acquisition_function = acquisition_function,
                              acq_fun_params = acq_fun_params, no_plots = no_plots)
            
            #results.append([optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step])
            results.append([BO_batch, x_next])
            
            optima.append(optimum)
            
            X_steps.append(X_step)
            Y_steps.append(Y_step)
            
            data_fusion_data_all.append(data_fusion_data)
            
            lengthscales_all.append(lengthscales)
            variances_all.append(variances)
            max_gradients.append(max_gradients)
            
            print('Repetition ', i)
 
        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
        filename_prefix = pickle_prefix + time_now + '/'
        if not os.path.exists(filename_prefix):
            os.makedirs(filename_prefix)
        filename_prefix = filename_prefix + time_now
        
        pickle_variables = [optima, X_steps, Y_steps, data_fusion_data_all,
                            results, lengthscales_all, variances_all, max_gradients_all]
        
        # Save the results as an backup
        for i in range(len(pickle_variable_names)):
            #print('Saving variable ', pickle_variable_names[i])
            filename = filename_prefix + '_' + pickle_variable_names[i] + pickle_postfix
            dbfile = open(filename, 'ab') 
            pickle.dump(pickle_variables[i], dbfile)                      
            dbfile.close()
        
    else:
        
        filename_prefix = pickle_prefix + fetch_file_date + '/' + fetch_file_date
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
        data_fusion_data_all = pickle_variables[3]
        results = pickle_variables[4]
        lengthscales_all = pickle_variables[5]
        variances_all = pickle_variables[6]
        max_gradients_all = pickle_variables[7]
        
    optima = np.array(optima)/60
    
    #mean = np.mean(optima)
    
    # Plot optimum vs BO rounds.
    
    cols = ['Optimum'+ x for x in list(map(str, range(n_rounds)))]
    df_optima_wide = pd.DataFrame(optima, columns = cols)#range(n_rounds))
    df_optima_wide['Repeat'] = range(n_repetitions) 
    df_optima_long = pd.wide_to_long(df_optima_wide, stubnames = 'Optimum', i = 'Repeat', j = 'Round')
    
    # Kuhunkin iteraation mennessä löydetty paras samplattu arvo plotattuna
    # Optimum y-arvo, paras oikeasti samplatty tiettyyn kierrokseen mennessä
    # Round on sama kuin sinun iter, esim 100
    # Repeat on se, kuinka monta kertaa BO-sykli toistetaan samoilla asetuksilla (statistiikka, std), esim 50
    plt.figure()
    sn.lineplot(data = df_optima_long, x = 'Round', y = 'Optimum', ci = 90, color = color)
    plt.ylim([0,10000])
    plt.tight_layout()
    
    if save_figs:
        
        plt.gcf().savefig(filename + '_optimum.pdf', transparent = True)
        plt.gcf().savefig(filename + '_optimum.svg', transparent = True)
        plt.gcf().savefig(filename + '_optimum.png', dpi=300)
    
    plt.show()
    
    # Plot "X distance" as the regret vs BO rounds
    
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
    sn.lineplot(data = df_regrets_long, x = 'Round', y = 'Regret', ci = 90, color = color)
    plt.ylim([0,1])
    plt.tight_layout()
    
    if save_figs:
    
        plt.gcf().savefig(filename + '_regret.pdf', transparent = True)
        plt.gcf().savefig(filename + '_regret.svg', transparent = True)
        plt.gcf().savefig(filename + '_regret.png', dpi=300)
    
    plt.show()
    
    if acquisition_function == 'EI_DFT':
        
        # Plot N_data_fusion_data vs BO_rounds.
        
        n_df_points = []
        # Repeats
        for i in range(len(data_fusion_data_all)):
            
            n_df_points.append([])
            
            # BO rounds, note that the last suggestion is always for the round
            # that has not yet been implemented.
            for j in range(len(data_fusion_data_all[i])-1):
                
                if j == 0:
                    n_df_points[i].append(data_fusion_data_all[i][j].shape[0])
                    
                else:
                    n_df_points[i].append(data_fusion_data_all[i][j].shape[0]+n_df_points[i][-1])
                
        cols = ['N_data_fusion_points'+ x for x in list(map(str, range(n_rounds)))]
        df_optima_wide = pd.DataFrame(np.array(n_df_points), columns = cols)#range(n_rounds))
        df_optima_wide['Repeat'] = range(n_repetitions) 
        df_optima_long = pd.wide_to_long(df_optima_wide, stubnames = 'N_data_fusion_points', i = 'Repeat', j = 'Round')
        
        plt.figure()
        sn.lineplot(data = df_optima_long, x = 'Round', y = 'N_data_fusion_points', ci = 90, color = color)
        #plt.ylim([0,450000/60])
        plt.tight_layout()
        
        if save_figs:
            
            plt.gcf().savefig(filename + '_Ndfpoints.pdf', transparent = True)
            plt.gcf().savefig(filename + '_Ndfpoints.svg', transparent = True)
            plt.gcf().savefig(filename + '_Ndfpoints.png', dpi=300)
        
        plt.show()
