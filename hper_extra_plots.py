#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:34:05 2022

Extra plots

@author: armi
"""
from hper_bo_simulated_with_df_function import bo_sim_target, ei_dft_param_builder, ei_dft_param2str, predict_points
from set_figure_defaults import FigureDefaults # C Antti Vepsalainen

import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import datetime
import matplotlib.pyplot as plt

def fetch_gt_model_reference(
        bo_ground_truth_model_path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222',
        ground_truth = [0.17, 0.03, 0.80],
        materials = ['CsPbI', 'MAPbI', 'FAPbI']):
    
    with open(bo_ground_truth_model_path,'rb') as f:
        gt_stability_model = pickle.load(f)

    ground_truth_y, ground_truth_y_std = predict_points(gt_stability_model, np.array([ground_truth]), Y_train = gt_stability_model.model.Y)

    ground_truth_y = ground_truth_y/60 #np.ravel(ground_truth_y-ground_truth_y_std)/60 # To px*h
    ground_truth_y_std = ground_truth_y_std/60
    
    # Relative noise is high in the ground truth area. Thus, BO that samples
    # the noisy predictions will commonly find optima that are clearly lower
    # than ground_truth_y. A more meaningful "ground truth value" is this:
    ground_truth_y_adj = (ground_truth_y - ground_truth_y_std)[0][0]


    return ground_truth_y, ground_truth_y_std, ground_truth_y_adj


def fetch_repetition_results(date, n_repetitions = 25, n_rounds = 100,
                             n_init = 2, batch_size = 1,
                             acquisition_function = 'EI_DFT',
                             data_fusion_property = None,
                             data_fusion_method = 'model_necessary'):

    # Typically, one does not need to modify these inputs.
    if acquisition_function == 'EI_DFT':
          acq_fun_params = [ei_dft_param_builder('EI_DFT',
              data_fusion_target_variable = data_fusion_property), 
              data_fusion_property, data_fusion_method]
          acq_fun_param_str = ei_dft_param2str(acq_fun_params)
          acquisition_description = (acquisition_function + acq_fun_param_str + 
                                     '_' + data_fusion_method)
    
    else:
        acquisition_description = acquisition_function
        acq_fun_params = [None, None, None]
        
    fetch_file_postfix = ('_nreps' + str(n_repetitions) + '_nrounds' + str(n_rounds) +
                  '_inits' + str(n_init) + '_batch' + str(batch_size) +
                  '_acq' + acquisition_description)

    pickle_prefix = ['./Results/', '/']
    pickle_variable_names = ['optima', 'X_accum', 'Y_accum', 'data_fusion_data', 'BOmainresults']
    pickle_postfix = fetch_file_postfix


    filename_prefix = pickle_prefix[0] + date + pickle_prefix[1] + date
    pickle_variables = []

    # Fetch the results from pickled backup
    for i in range(len(pickle_variable_names)):
        filename = filename_prefix + '_' + pickle_variable_names[i] + pickle_postfix
        dbfile = open(filename, 'rb') 
        pickle_variables.append(pickle.load(dbfile))                      
        dbfile.close()
        
    optima = pickle_variables[0]
    X_accum = pickle_variables[1]
    Y_accum = pickle_variables[2]
    data_fusion_data_all = pickle_variables[3]
    results = pickle_variables[4]
    
    
    optima = np.array(optima)/60 # To px*h
    
    return optima, X_accum, Y_accum, data_fusion_data_all, results

def format_df_for_plots(quantity, label, n_repetitions = 25, n_rounds = 100):
    
    cols = [label + x for x in list(map(str, range(n_rounds)))]
    df_wide = pd.DataFrame(quantity, columns = cols)#range(n_rounds))
    df_wide['Repeat'] = range(n_repetitions) 
    df_long = pd.wide_to_long(df_wide, stubnames = label, i = 'Repeat', j = 'Round')

    return df_wide, df_long

'Optimum $I_c$ (px h)'

def format_df_optima(optima, label = 'Optimum $I_c$ (px h)', n_repetitions = 25,
                     n_rounds = 100):
    
    df_optima_wide, df_optima_long = format_df_for_plots(optima, label, 
                                           n_repetitions = n_repetitions,
                                           n_rounds = n_rounds)
    
    return df_optima_wide, df_optima_long

def format_df_regrets(X_accum, Y_accum, n_repetitions = 25,
                     n_rounds = 100, regret_shift = 0, ground_truth = [0.17, 0.03, 0.80]):
    
    # The optimum region in the gt_stability_model is flat. Its width is appr.
    # 0.06, i.e. the optimum region has been found if the regret is below 0.03.
    # Actually, 0.06.
    #regret_shift = 0#0.06 # Will be used below.
    
    regrets = [[None for i in range(n_rounds)] for i in range(n_repetitions)]

    for i in range(n_repetitions):
    
        Y_accum_all = Y_accum[i]#results[i][9]
        X_accum_all = X_accum[i]#results[i][10]
    
        for j in range(n_rounds):
    
            idx_optimum = np.argmin(Y_step_all[j])
            X_optimum = X_step_all[j][idx_optimum]
            regret = np.sqrt(np.sum((ground_truth - X_optimum)**2))
            regrets[i][j] = regret - regret_shift
            
        df_regrets_wide, df_regrets_long = format_df_for_plots(regrets, 'Regret', 
                                            n_repetitions = n_repetitions,
                                            n_rounds = n_rounds)
        
        return df_regrets_wide, df_regrets_long
 
def format_df_dfpoints(data_fusion_data_all, n_repetitions = 25,
                     n_rounds = 100):


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


    print('Number of data fusion points on average: ', np.mean(np.array(n_df_points), axis=0)[-1])
    
    df_datafusion_wide, df_datafusion_long = format_df_for_plots(n_df_points, 'Human queries', 
                                        n_repetitions = n_repetitions,
                                        n_rounds = n_rounds)
    
    return df_datafusion_wide, df_datafusion_long

#####################
sn.set_context('paper')
to_same_plot = True
figsize = [3, 2]

if to_same_plot:
    
    fo = plt.figure() # Optimum figure
    fr = plt.figure() # Regret figure
    fn = plt.figure() # N data fusion points figure
    legend_list = [] # List of items in the legend, is filled in the loop below.


# Set figure style.
mystyle = FigureDefaults('nature_comp_mat_sc')
####################


# Regret round 100 from different runs.

fetch_file_dates = ['202210171746', '202209292216', '202210181625',
                    '202210190550']

labels = ['r=0.1, $\sigma$/l^0.5', 'r=0.1, $\sigma$/l^0.5/2', 'r=0.1, $\sigma$/l^0.5/4',
          'r=0.1, $\sigma$/l^0.5/8', 'r=0.025, $\sigma$/l^0.5/4']

ground_truth_y, ground_truth_y_std, ground_truth_y_adj = fetch_gt_model_reference()


all_df_regrets_wide = []
all_df_regrets_long = []
all_res = []

for i in range(len(fetch_file_dates)):
    
    date = fetch_file_dates[i]
    
    optima, X_accum, Y_accum, data_fusion_data_all, results = fetch_repetition_results(date, data_fusion_property = 'visual')
    
    df_regrets_wide, df_regrets_long = format_df_regrets(X_accum, Y_accum)
    
    all_res.append([optima, X_accum, Y_accum, data_fusion_data_all, results])
    
    all_df_regrets_wide.append(df_regrets_wide)
    all_df_regrets_long.append(df_regrets_long)
    
if to_same_plot:
    current_fig = fr
else:
    current_fig = plt.figure()


m = []
s = []

for i in range(len(fetch_file_dates)):
    
    m.append(all_df_regrets_wide[i]['Regret99'].mean())
    
    s.append(all_df_regrets_wide[i]['Regret99'].std())
    
plt.figure()
plt.plot([1, 0.5, 0.25, 0.125], m, 'x')

    
'''
plt.figure(current_fig.number)
plt.plot()
#sn.lineplot(data = df_regrets_long, x = 'Round', y = 'Regret', ci = 90, color = color)
plt.ylim([0,1.2])
plt.gcf().set_size_inches(figsize)
plt.tight_layout()
'''  