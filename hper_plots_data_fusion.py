#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import datetime

from hper_plots_target import init_plots, plot_mean_and_data, plot_std_only, plot_acq_only, fill_ternary_grid
import GPyOpt

def fill_ternary_grids(mean, std, p, rounds, df_models, points, p_beta, p_midpoint, y_train_data = None):
    
    for i in range(rounds):
        
        if y_train_data is None:
            y_t = None
        else:
            y_t = y_train_data[i]
        
        if df_models[i] is None: # Data fusion set but none samples have been collected (yet).
            
            # Set empty value for plots.
            mean[i] = np.zeros((points.shape[0],1))
            std[i] = np.zeros((points.shape[0],1))
            p[i] = np.zeros((points.shape[0],1)) + 0.5
        
        else:
            
            # : Here the posterior mean and std_dv+acquisition function are calculated.
            mean[i], std[i] = fill_ternary_grid(mean[i], std[i], df_models[i], points, y_train_data = y_t)
        
            p[i] = GPyOpt.acquisitions.EI_DF.calc_P(points, df_models[i], 
                                                       p_beta = p_beta, 
                                                       p_midpoint = p_midpoint)
            
    return mean, std, p

def plotDF(rounds, materials, df_models, df_data_accum, df_variable,
           #lengthscale, variance, noise_variance, 
           p_beta, p_midpoint, limit_file_number = True,
           time_str = None, results_folder = './Results/', close_figs = False):
    
           
    #suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step, limit_file_number = True, time_str = None, new_df_points_x, new_df_points_y, current_data_fusion_data, current_df_model, beta, midpoint):
    points, gp_mean, gp_std, p, _, time_now, results_dir, rounds_to_plot = init_plots(
        rounds, limit_file_number, time_str, results_folder)
    
    '''
    # Collect data fusion data accumulated by each round.
    df_data_cumulative = []
    for i in range(rounds):
                
        if i == 0:
            df_data_cumulative.append(df_data[i].copy())
            
        else:
            df_data_cumulative.append(pd.concat([df_data[i], 
                                                 df_data_cumulative[-1]],
                                                ignore_index = True))
        if df_models[i] is None:
            
            Exception('The current implementation of plotting data fusion requires a model. Provide data fusion models.')
     
            # Train model.
            df_models[i] = GP_model(df_data_cumulative[i],
                                    data_fusion_target_variable = df_variable,
                                    lengthscale = lengthscale, 
                                    variance = variance,
                                    noise_variance = None,
                                    materials)
            
            current_df_model = GP_model(data_fusion_XZ_accum[k],
                                        data_fusion_target_variable = acq_fun_params['df_target_var'],
                                        lengthscale = acq_fun_params['gp_lengthscale'],
                                        variance = acq_fun_params['gp_variance'],
                                        noise_variance = None,
                                        data_fusion_input_variables = acq_fun_params['df_input_var'])
    '''

    # Fill in the lists with surfaces to plot.
    gp_mean, gp_std, p = fill_ternary_grids(gp_mean, gp_std, p, rounds, df_models,
                                            points, p_beta, p_midpoint)
    
        
    ###############################################################################
    # PLOT
    ###############################################################################
    # Let's plot the data fusion P, mean, and st.devs. This plot works for 3 materials only.
    
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    axis_scale = 1
    # These could be set based on data but then contour levels (and their ticks
    # are hard to calculate.
    lims_m = [np.min(gp_mean)/axis_scale, np.max(gp_mean)/axis_scale]
    lims_s = [np.min(gp_std)/axis_scale, np.max(gp_std)/axis_scale]
    lims_p = [0,1]
    
    for i in rounds_to_plot:
        
        data_x = df_data_accum[i].loc[:,materials].values
        data_y = df_data_accum[i].iloc[:,-1].values/axis_scale
        
        if data_x.shape[0] == 0:
            
            # No data for this round.
            data_x = None
            data_y = None
        
        # Plot posterior mean with samples acquired by that round.
        plot_mean_and_data(points, gp_mean[i]/axis_scale,
                               data_x, 
                               data_y,
                               color_lims = lims_m,
                               cbar_label =
                               r'Human in round ' + str(i),
                               saveas = results_dir +
                               'DF-round' + str(i) + '-' + time_now,
                               #cbar_ticks = np.arange(-1.5, 1.51, 0.5)
                               close_figs = close_figs
                               )
            
        plot_std_only(points, gp_std[i]/axis_scale,
                          color_lims = lims_s,
                          cbar_label = r'Std. of human in round' + str(i),
                          saveas = results_dir + 'St-Dev-of-df-round' +
                              str(i) + '-' + time_now,
                              #cbar_ticks = np.arange(-1.5, 1.51, 0.5)
                              close_figs = close_figs
                              )
                
        plot_acq_only(points, p[i], color_lims = lims_p,
                          cbar_label = r'P(high quality) in round' + str(i),
                          saveas = results_dir + 'P-round'+str(i) + 
                                         '-' + time_now, close_figs = close_figs)
                
    plt.close('all')
        