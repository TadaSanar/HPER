#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:23 2022

@author: armi
"""

from hper_bo_noiselevel import bo_sim_target, acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr, predict_points
from hper_repetitions_noiselevel import build_filenames, cg, p_above
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#%load_ext autoreload
#%autoreload 2

# TO DO: Fix, it is not good to use both sn and figuredefaults, and figsize.
#from set_figure_defaults import FigureDefaults # C Antti Vepsalainen
# Set figure style.
#mystyle = FigureDefaults('nature_comp_mat_sc')

sn.set_context('paper')

cmap = mpl.cm.viridis
cmap.set_under(color = 'r')



def plot_ho(regrets_interm_mean, n_humans_mean, c_eigs, c_exclzs, c_grads, jitters, n_eig,
            n_exclz, bo_params, plot_proportions = True, share_range = False):

    plt.figure()
    plt.plot(regrets_interm_mean)
    plt.show()
    
    # EIG repeats, excl. zone repeats
    #n_total_ho = n_eig + n_exclz
    
    # Additionally, vanilla and always human are also assumed to have been
    # tested for each jitter value.
    
    # Plot P_above instead of c_grad.
    p_aboves = p_above(c_grads)
    x_data = p_aboves
    x_label = '$P_{above\,limit}$'
    
    # Number of human samples are determined by param n_init in the first round
    # of bo. After that, max 1 human sample is collected in each round (if no
    # batch mode). 
    n_human_max = (bo_params['n_init'] + (bo_params['n_rounds'] - 1) *
                   bo_params['batch_size'])
    
    # Plot the number of human samples as proportions.
    if plot_proportions is True:
        
        z_data_human = np.array(n_humans_mean)/n_human_max
        z_label_human = '$\overline{N}_{human}/N_{human,\,max}$'
        # Init samples proportion of all human samples (to be highlighted in the plot)
        n_human_min = bo_params['n_init']/n_human_max
        step = 0.01
        cbar_ticks_human = np.linspace(n_human_min + step, 1,5)
        cbar_format_human = '%.2f'
        
        if share_range is True:
            
            vmax_human = cbar_ticks_human[-1]
        
        else:
            
            vmax_human = None
        
    else:
        
        z_data_human = n_humans_mean
        z_label_human = '$\overline{N}_{human}$'
        n_human_min = bo_params['n_init']
        cbar_ticks_human = np.linspace(n_human_min + step, n_human_max,5)
        cbar_format_human = '%.0f'
        
    z_label_regret = 'Mean regret (px$\cdot$h)'
    cbar_format_regret = '%.2f'
    
    for j in range(len(jitters)):
        
        filename_prefix = 'HO_jitter' + str(jitters[j])
        
        idx_start = int(j * (len(regrets_interm_mean) / len(jitters)))
        idx_end = int((j+1) * (len(regrets_interm_mean) / len(jitters)))
        #idx_start = j * n_total_ho
        #idx_end = idx_start + n_eig
        filename_postfix = '_eig_regret'
        
        #cbar_ticks_regret = np.linspace(np.min(regrets_interm_mean[idx_start:idx_end]), 
        #                                np.max(regrets_interm_mean[idx_start:idx_end]), 5)
        
        # Plot EIG. Methods that are not EIG will not be visible in the plot
        # because they have np.nan values.
        plt.figure()
        plt.scatter(x_data[idx_start:idx_end], c_eigs[idx_start:idx_end], 
                    c = regrets_interm_mean[idx_start:idx_end])#,
        #            vmin = cbar_ticks_regret[0],
        #            vmax = cbar_ticks_regret[-1])
        #plt.scatter(x_data[idx_start:idx_end], c_es[idx_start:idx_end], 
        #            c = regrets_interm_mean[(idx_start + (2*(j+1))):(idx_end + (2*(j+1)))])
        plt.colorbar(#ticks = cbar_ticks_regret, 
                     label = z_label_regret,
                     format = cbar_format_regret)
        plt.title('Regret after ' + str(regret_interm_idx+1) + 
                  ' rounds (DF-I method, jitter ' + 
                  str(jitters[j]) +')')
        plt.xlabel(x_label)
        plt.ylabel('c_eig')
        
        if save_figs:
            
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.pdf', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.svg', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.png', dpi=300)
        
        plt.show()
        
        filename_postfix = '_eig_nhumans'
        plt.figure()
        plt.scatter(x_data[idx_start:idx_end], c_eigs[idx_start:idx_end], 
                    c = z_data_human[idx_start:idx_end],
                    vmin = n_human_min + step,
                    vmax = vmax_human)
        #plt.scatter(x_data[idx_start:idx_end], z_data_human[idx_start:idx_end], 
        #            c = n_humans_mean[(idx_start + (j+1)):(idx_end + (j+1))],
        #            vmin = 3.01)
        
        if share_range is True:
            
            plt.colorbar(ticks = cbar_ticks_human, extend='min', label = z_label_human,
                     format = cbar_format_human)
            
        else:
            
            plt.colorbar(extend='min', label = z_label_human,
                     format = cbar_format_human)
            
        plt.title('$N_{humans}$ (DF-I method, jitter ' + 
                  str(jitters[j]) +')')
        plt.xlabel(x_label)
        plt.ylabel('c_eig')
        
        if save_figs:
            
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.pdf', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.svg', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.png', dpi=300)
        
        plt.show()
        
        filename_postfix = '_exclz_regret'
        #idx_start = (j * n_total_ho) + n_eig
        #idx_end = idx_start + n_exclz
        plt.figure()
        plt.scatter(x_data[idx_start:idx_end], c_exclzs[idx_start:idx_end], 
                    c = regrets_interm_mean[idx_start:idx_end])#,
                    #vmin = cbar_ticks_regret[0],
                    #vmax = cbar_ticks_regret[-1])
        #plt.scatter(x_data[idx_start:idx_end], c_es[idx_start:idx_end], 
        #            c = regrets_interm_mean[(idx_start + (2*(j+1))):(idx_end + (2*(j+1)))])
        plt.colorbar(#ticks = cbar_ticks_regret, 
                     label = z_label_regret,
                     format = cbar_format_regret)
        plt.title('Regret after ' + str(regret_interm_idx+1) + 
                  ' rounds (DF-Z method, jitter ' + 
                  str(jitters[j]) +')')
        plt.xlabel(x_label)
        plt.ylabel('c_exclz')
        
        if save_figs:
            
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.pdf', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.svg', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.png', dpi=300)
        
        plt.show()
        
        filename_postfix = '_exclz_nhumans'
        plt.figure()
        plt.scatter(x_data[idx_start:idx_end], c_exclzs[idx_start:idx_end], 
                    c = z_data_human[idx_start:idx_end],
                    vmin = n_human_min + step,
                    vmax = vmax_human)
        #plt.scatter(x_data[idx_start:idx_end], c_es[idx_start:idx_end], 
        #            c = n_humans_mean[(idx_start + (j+1)):(idx_end + (j+1))],
        #            vmin = 3.01)
        if share_range is True:
            
            plt.colorbar(ticks = cbar_ticks_human, extend='min', label = z_label_human,
                     format = cbar_format_human)
            
        else:
            
            plt.colorbar(extend='min', label = z_label_human,
                     format = cbar_format_human)
            
        plt.title('$N_{humans}$ (DF-Z method, jitter ' + 
                  str(jitters[j]) +')')
        plt.xlabel(x_label)
        plt.ylabel('c_exclz')
        
        if save_figs:
            
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.pdf', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.svg', transparent = True)
            plt.gcf().savefig(folder + filename_prefix + filename_postfix + '.png', dpi=300)
        
        plt.show()
        

def create_wide_and_long_df(values, var_name, n_rounds, n_repetitions):
    '''
    Format the data in the way that seaborn understands.
    '''
    
    cols = [var_name + x for x in list(map(str, range(n_rounds)))]
    df_wide = pd.DataFrame(values, columns = cols)
    df_wide['Repeat'] = range(n_repetitions) 
    df_long = pd.wide_to_long(df_wide, stubnames = var_name, i = 'Repeat',
                              j = 'Round')
    
    return df_wide, df_long

def create_plot(df_data_long, name_in_legend, color, legend_list,
                figsize = [3,2], fig_handle = None,
                ground_truth_reference = None, ylim = None, ylog = False, 
                xlim = None):

    if fig_handle is None:
        current_fig = plt.figure()
    else:
        current_fig = fig_handle
    
    plt.figure(current_fig.number)
    ax = plt.axes()
    
    
    # Mean values
    sn.lineplot(data = df_data_long, x = 'Round', y = df_data_long.columns[0],
                ci = 90, color = color, ax = ax)
    legend_list.append(name_in_legend)
    
    if (ylim is not None) and (ylog is False):
     	plt.ylim([0,ylim])
    elif (ylim is not None) and (ylog is True):
        plt.ylim([1, ylim])
    
    if (xlim is not None):
     	plt.xlim([0,xlim])
    
    
    #plt.xlim([0,50])
    
    if ground_truth_reference is not None:
        # Ground truth reference
        n_rounds = np.max(df_data_long.index.get_level_values('Round'))
        plt.plot([0,n_rounds], [ground_truth_reference, ground_truth_reference],
              '--k', linewidth=0.5)
        legend_list.append('Ground truth region')
        
    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()
    
    if fig_handle is None:
        
        # Add Std into the legend list only if this is the only plot coming to
        # the figure. Otherwise, add it only when the figure is saved (as
        # seaborn groups all std lines to the end).
        legend_list.append('Std')

        if legends_visible:
                plt.legend(legend_list)
                legend_list = [] # Wipe for the next figure.
        
    # If fig_handle is not None: Multiple plots will be placed into the same fig,
    # Therefore legend are created only at the end of the loop.
    
    return current_fig, legend_list

def create_barplot(data_mean, data_std, name_in_legend, color,
                figsize = [3,2], ylabel = None, fig_handle = None,
                ground_truth_reference = None, ylim = 1.00, figname = 'fig'):

    if fig_handle is None:
        current_fig = plt.figure()
    else:
        current_fig = fig_handle
    
    plt.figure(current_fig.number)
    
    ax = plt.axes()
    
    #bar = series_boolean.astype(int).mean()
    
    x = ['Not', 'Always', 'When\nnecessary']
    y = data_mean
    e = data_std
    
    #sn.color_palette()[1]
    # Mean values
    sn.barplot(x, y, ax = ax, label = name_in_legend)
    #legend_list.append(name_in_legend)
    
    plt.ylabel(ylabel)
    plt.xlabel('Human queried')
    #if (ylim is not None):
    #plt.title('Searches that converge into Region A')
    #plt.ylim([0,ylim])
        
    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()
    
    if fig_handle is not None:
        
        if legends_visible:
                plt.legend(legend_list)
    
    plt.gcf().savefig(figname + '.pdf', transparent = True)
    plt.gcf().savefig(figname + '.svg', transparent = True)
    plt.gcf().savefig(figname + '.png', dpi=300)
    
    return current_fig, legend_list




            

###############################################################################

c_eig = [0.95]#[0, 0.2, 0.5, 0.8, 1]#[0, 0.5, 0.75, 0.9, 1, 2]  # Expected information gain.
# Size of the exclusion zone in percentage points (max. 100)
c_exclz = [15]#[0,1,5,10,20]#[1, 5, 10, 20]
# Gradient limit. 0.05#, 0.07, 0.1, 0.2, 0.5, 0.75
c_g = list(cg(np.array([0.8])))#0.2, 0.5, 0.8, 1])))#list(cg(np.array([0.2, 0.5, 0.6, 0.8, 1])))

hyperparams_eig = []
hyperparams_exclz = []
for i in range(len(c_g)):
    for j in range(len(c_exclz)):

        hyperparams_exclz.append((c_g[i], c_exclz[j]))

    for j in range(len(c_eig)):

        hyperparams_eig.append((c_g[i], c_eig[j]))
        
jitters = [0.01]#, 0.1]


n_eig  = len(hyperparams_eig)
n_exclz  = len(hyperparams_exclz)
n_hpars = 2 + n_eig + n_exclz
n_j = len(jitters)

folder = './Results/20240409-comp-to-boss-no-noise-add-constr-noard-nonorm-Mat52kernel/'#'20231129-noisytarget-noisyhuman-ho-j001/'#'./Results/triton/20230823-noisytarget-noisyhuman-ho/'
#ground_truth_rA = [0.17, 0.03, 0.80]  # From C2a paper
ground_truth_rA = [0.165, 0.04, 0.79]  # From the ground truth model

bo_params = {'n_repetitions': 25,
             'n_rounds': 40,#75,
             'n_init': 3,
             'batch_size': 1,
             'materials': ['CsPbI', 'MAPbI', 'FAPbI'],
             'noise_target': 0
             }

noise_df = 0#1

ground_truth_rB = [1,0,0]  # Ground truth location of Region B

###############################################################################

# Give True if you don't want to run new BO but only fetch old results and re-plot them.
fetch_old_results = True
# Give False if you don't want to save the figures.
save_figs = True
# Set legends off if you want to see the figures better.
legends_visible = False

# When plotting HO results, it may be beneficial to plot intermediate regret
# values if the searches have all converged. Insert the round you want to plot
# int his case.
regret_interm_idx = 39#34#19

bo_ground_truth_model_path = './Source_data/stability_model_GPyHomoscedastic'#'./Source_data/stability_model_improved_region_B.p'#
to_same_plot = True
folder_list = np.sort(next(os.walk(folder), (None, [], None))[1])

# Assumes that all the BO results folders start with digit, the rest are added
# by the user and should not be considered.
while folder_list[-1][0].isdigit() is False:
    
    folder_list = folder_list[0:-1]

dates_list = [folder_list[i][0:12] for i in range(len(folder_list))]

###############################################################################

materials = bo_params['materials']
n_rounds = bo_params['n_rounds']
n_repetitions = bo_params['n_repetitions']

legend_list = [[], [], [], []] # List of items in the legend, is filled in the loop below.

if to_same_plot:
    
    fo = plt.figure() # Optimum figure
    fr = plt.figure() # Regret figure
    fn = plt.figure() # N data fusion points figure
    fb = plt.figure() # Simple bar plot on regions A/B

else:
    
    fo = None
    fr = None
    fn = None
    fb = None
    
# Fetch ground truth model and optimum y region.

with open(bo_ground_truth_model_path,'rb') as f:
    gt_stability_model = pickle.load(f)
    #gt_stability_model = gt_stability_model.model # GPy GPregression model if the loaded model was GPyOpt GPModel

ground_truth_y, ground_truth_y_var = predict_points(gt_stability_model,
                                                    np.array([ground_truth_rA]),
                                                    Y_data = gt_stability_model.Y)

ground_truth_y = ground_truth_y/60 #np.ravel(ground_truth_y-ground_truth_y_std)/60 # To px*h
ground_truth_y_std = ground_truth_y_var/60

# Relative noise is high in the ground truth area. Thus, BO that samples
# the noisy predictions will commonly find optima that are clearly lower
# than ground_truth_y. A more meaningful "ground truth value" is this:
ground_truth_y_adj = ground_truth_y[0][0]#(ground_truth_y - ground_truth_y_std)[0][0]

###############################################################################

# Viikon

regrets_interm_mean = []
regrets_interm_std = []
c_grads = []
c_eigs = []
c_exclzs = []
n_humans_mean = []

for m in range(len(dates_list)):

    if (m > -1):

        if (m % n_hpars) == 0:

            data_fusion_property = None
            acquisition_function = 'EI'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[0]
            name_in_legend = 'Vanilla'
            
        elif (m % n_hpars) == 1:

            data_fusion_property = 'visual'
            df_data_coll_method = 'model_all'
            acquisition_function = 'EI_DFT'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[2]
            name_in_legend = 'Human always'

        elif (m % n_hpars) < (n_hpars - n_exclz):
            
            data_fusion_property = 'visual'
            df_data_coll_method = 'model_necessary_eig'
            c_grad = hyperparams_eig[(m % n_hpars)-2][0]
            c_e = hyperparams_eig[(m % n_hpars)-2][1]
            acquisition_function = 'EI_DFT'
            color = sn.color_palette()[1]
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            name_in_legend = 'EIG'
        
        else:

            data_fusion_property = 'visual'
            df_data_coll_method = 'model_necessary_exclz'
            c_grad = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][0]
            c_e = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][1]
            acquisition_function = 'EI_DFT'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[3]
            name_in_legend = 'Excl. zone'

        jitter = jitters[m // n_hpars]


        fetch_file_date = dates_list[m]
        ###############
        # Typically, one does not need to modify these inputs.
        
        acq_fun_params = acq_param_builder(acquisition_function,
                                          data_fusion_property = data_fusion_property,
                                          data_fusion_input_variables = bo_params['materials'],
                                          optional_acq_params = {'jitter': jitter})
        acq_fun_descr = acq_fun_param2descr(acquisition_function, acq_fun_params = acq_fun_params)
        
        
        if data_fusion_property is None:

            df_data_coll_params = df_data_coll_param_builder()

        elif (df_data_coll_method == 'model_all') or (df_data_coll_method == 'model_none'):

            df_data_coll_params = df_data_coll_param_builder(
                df_method=df_data_coll_method, noise_df = noise_df)

        else:

            df_data_coll_params = df_data_coll_param_builder(df_method=df_data_coll_method,
                                                             gradient_param=c_grad,
                                                             exclusion_param=c_e,
                                                             noise_df = noise_df)

        df_data_coll_descr = df_data_coll_method_param2descr(df_data_coll_params)
        
        pickle_filenames, figure_filenames, t_folder = build_filenames(folder, bo_params,
                                                             acq_fun_descr,
                                                             df_data_coll_descr,
                                                             fetch_file_date = fetch_file_date,
                                                             m = m)

        # Set figure style.
        #mystyle = FigureDefaults('nature_comp_mat_sc')

        ###############

        all_starting_points = []
        results = []
        optima = []
        X_accum = []
        Y_accum = []
        data_fusion_data_all = []
        lengthscales_all = []
        variances_all = []
        max_gradients_all = []

        ###############################################################################
        
        # Fetch old results.
        
        pickle_variables = []

        # Fetch the results from pickled backup
        for s in pickle_filenames:
            dbfile = open(s, 'rb')
            pickle_variables.append(pickle.load(dbfile))
            dbfile.close()

        optima = pickle_variables[0]
        X_accum = pickle_variables[1]
        Y_accum = pickle_variables[2]
        data_fusion_data_all = pickle_variables[3]
        results = pickle_variables[4]
        lengthscales_all = pickle_variables[5]
        variances_all = pickle_variables[6]
        max_gradients_all = pickle_variables[7]

        optima = np.array(optima)/60 # To px*h
        
        #######################################################################
        # Check that all the datapoints are inside the triangle. A plot and
        # printouts are created if any of them are outside.
        #test_outside_triangle(X_accum)
        
        #######################################################################
        
        # Plot optimum vs BO rounds.
        df_optima_wide, df_optima_long = create_wide_and_long_df(optima,
                                                                 'Optimum $I_c$ (px$\cdot$h)',
                                                                 n_rounds,
                                                                 n_repetitions)
        if to_same_plot and (len(legend_list[0])>0):
            # Do not plot ref line more than once into the same plot.
            ground_truth_reference = None
        else:
            ground_truth_reference = ground_truth_y_adj
        
        current_fig, legend_list[0] = create_plot(df_optima_long, name_in_legend,
                                               color, legend_list[0],
                                               fig_handle = fo,
                                               ground_truth_reference=ground_truth_reference,
                                               ylim = 10000, ylog = False)
        
        if (to_same_plot is False) and save_figs:
            
            plt.gcf().savefig(figure_filenames[0] + '.pdf', transparent = True)
            plt.gcf().savefig(figure_filenames[0] + '.svg', transparent = True)
            plt.gcf().savefig(figure_filenames[0] + '.png', dpi=300)
    
        #################################
        # Plot "X distance" as the regret vs BO rounds
        
        # The optimum region in the gt_stability_model is flat. Its width is appr.
        # 0.06, i.e. the optimum region has been found if the regret is below 0.03.
        # Actually, 0.06.
        #regret_shift = 0.06 # Will be used below.
    
        regrets = [[None for i in range(n_rounds)] for i in range(n_repetitions)]
        regrets_rB = [[None for i in range(n_rounds)] for i in range(n_repetitions)]
        n_samples_rB = [None for i in range(n_repetitions)]
        
        for i in range(n_repetitions):
        
            Y_accum_all = Y_accum[i]
            X_accum_all = X_accum[i]
        
            for j in range(n_rounds):
        
                idx_optimum = np.argmin(Y_accum_all[j])
                X_optimum = X_accum_all[j][idx_optimum]
                regret = np.sqrt(np.sum((ground_truth_rA - X_optimum)**2))
                regrets[i][j] = regret# - regret_shift
                
                regret_rB = np.sqrt(np.sum((ground_truth_rB - X_optimum)**2))
                regrets_rB[i][j] = regret_rB
            
            n_samples_rB[i] = np.sum(np.sqrt(np.sum((ground_truth_rB - 
                                                     X_accum_all[regret_interm_idx])**2, 
                                                    axis = 1)) < 0.3)
                
            
                
        df_regrets_wide, df_regrets_long = create_wide_and_long_df(regrets,
                                                                 'Regret',
                                                                 n_rounds,
                                                                 n_repetitions)
        
        df_regrets_rB_wide, df_regrets_rB_long = create_wide_and_long_df(regrets_rB,
                                                                 'Regret_rB',
                                                                 n_rounds,
                                                                 n_repetitions)
        if to_same_plot and (len(legend_list[1])>0):
            # Do not plot ref line more than once into the same plot.
            ground_truth_reference = None
        else:
            ground_truth_reference = 0.15
    
        current_fig, legend_list[1] = create_plot(df_regrets_long, name_in_legend,
                                               color, legend_list[1],
                                               fig_handle = fr,
                                               ground_truth_reference=ground_truth_reference,
                                               ylim = 1.2, ylog = False,
                                               xlim = regret_interm_idx)
        
        if (to_same_plot is False) and save_figs:
            
            plt.gcf().savefig(figure_filenames[1] + '.pdf', transparent = True)
            plt.gcf().savefig(figure_filenames[1] + '.svg', transparent = True)
            plt.gcf().savefig(figure_filenames[1] + '.png', dpi=300)
        
        print('\nFinal regret: ' + "{0:.2f}".format(df_regrets_wide.iloc[:,-2].mean()))
        
        final_regret_rB = df_regrets_rB_wide.iloc[:, -2]
        final_regret_rA = df_regrets_wide.iloc[:, -2]
        # Optimum should be at least r=0.25 distance further away to Region B than Region A.
        rA_found = final_regret_rA + 0.1875 < final_regret_rB
        
        print(rA_found.astype(int).mean())
        print(rA_found.astype(int).std())
        
        print('Number of samples from region B: ', np.mean(n_samples_rB))
        n_samples = X_accum_all[regret_interm_idx].shape[0]
        prop_samples_rB_mean = np.mean(n_samples_rB)/n_samples
        prop_samples_rB_std = np.std(n_samples_rB)/n_samples
        print('Proportion of samples from region B: ', 
              prop_samples_rB_mean, prop_samples_rB_std)
        
                
        regrets_interm_mean.append(np.mean(
            df_regrets_wide['Regret' + str(regret_interm_idx)]))
        regrets_interm_std.append(np.std(
            df_regrets_wide['Regret' + str(regret_interm_idx)]))
        
        print('Regret after ' + str(regret_interm_idx+1) + ' rounds: ', 
              regrets_interm_mean[-1])
        
        if ((m % n_hpars) == 0) or ((m % n_hpars) == 1):
            
            # Vanilla or human always
            
            c_grads.append(np.nan)
            c_eigs.append(np.nan)
            c_exclzs.append(np.nan)
            
        elif ((m % n_hpars) < (n_hpars - n_exclz)):
            
            # EIG
            
            c_grads.append(c_grad)
            c_eigs.append(c_e)
            c_exclzs.append(np.nan)
            
            print('m:', m, ', c_g: ', c_grad, 'p_above: ', p_above(c_grad), 'c_eig: ', c_e)
            
        else:
            
            # Excl. zone
            
            c_grads.append(c_grad)
            c_eigs.append(np.nan)
            c_exclzs.append(c_e)
            
            print('m:', m, ', c_g: ', c_grad, 'p_above: ', p_above(c_grad), 'c_exclz: ', c_e)
            
        
        ######################################
        # Plot N_data_fusion_data vs BO_rounds.
        
        if acquisition_function == 'EI_DFT':
            
            n_df_points = []
            # Repeats
            for i in range(len(data_fusion_data_all)):
                
                n_df_points.append([])
                
                # BO rounds.
                for j in range(len(data_fusion_data_all[i])):
                    
                    if j == 0:
                        n_df_points[i].append(data_fusion_data_all[i][j].shape[0])
                        
                    else:
                        n_df_points[i].append(data_fusion_data_all[i][j].shape[0]+n_df_points[i][-1])
            
            
            print('Number of data fusion points on average for ' + 
                  name_in_legend + ' (date: ' + fetch_file_date + '): ', 
                  np.mean(np.array(n_df_points), axis=0)[-1])
            print('Proportion of data fusion points on average: ', 
                  np.mean(np.array(n_df_points), axis=0)[-1]/n_samples)
            df_datafusion_wide, df_datafusion_long = create_wide_and_long_df(n_df_points,
                                                                     'Human queries',
                                                                     n_rounds,
                                                                     n_repetitions)
    
            current_fig, legend_list[2] = create_plot(df_datafusion_long, name_in_legend,
                                                   color, legend_list[2],
                                                   fig_handle = fn,
                                                   ground_truth_reference=None,
                                                   ylim = 100, ylog = True)
            
            if (to_same_plot is False) and save_figs:
                
                plt.gcf().savefig(figure_filenames[2] + '.pdf', transparent = True)
                plt.gcf().savefig(figure_filenames[2] + '.svg', transparent = True)
                plt.gcf().savefig(figure_filenames[2] + '.png', dpi=300)
            
            n_humans_mean.append(np.mean(df_datafusion_wide.iloc[:,-2]))
            
        else:
            
            n_humans_mean.append(np.nan)
            
        

####################
# Save plots if not done before.

if to_same_plot:

    fig_handles = [fo, fr, fn]
    plotname_postfixes = ['_optimum', '_regret', '_N_df_human']

    for k in range(len(fig_handles)):

        current_fig = fig_handles[k]
        plt.figure(current_fig.number)
 
        if legends_visible:
            legend_list[k].append('Std')
            plt.legend(legend_list[k])
    
        if k == 2:
            
            for i in current_fig.axes:
                i.set(yscale="log")
        
        if save_figs:
            
            plt.gcf().savefig(folder + plotname_postfixes[k] + '.pdf', transparent = True)
            plt.gcf().savefig(folder + plotname_postfixes[k] + '.svg', transparent = True)
            plt.gcf().savefig(folder + plotname_postfixes[k] + '.png', dpi=300)

####################
# Show plots
plt.show()

plot_ho(regrets_interm_mean, 
        n_humans_mean, c_eigs, c_exclzs, c_grads, jitters, len(hyperparams_eig), 
        len(hyperparams_exclz), bo_params)



#current_fig, legend_list[3] = create_barplot(rA_found, name_in_legend,
#                                             color,
#                                             fig_handle = fb)
prop_samples_rB_mean = [0.173,0.004,0.006] # 150 rounds
prop_samples_rB_std = [0.2919,0.012,0.006] # 150 rounds

prop_samples_rB_mean = [0.173,0.005,0.02] # 100 rounds
prop_samples_rB_std = [0.276,0.012,0.01] # 100 rounds
current_fig, legend_list[3] = create_barplot(prop_samples_rB_mean, 
                                             prop_samples_rB_std,
                                             name_in_legend,
                                             color,
                                             fig_handle = fb,
                                             ylabel = r'$N_{low}$ $_{quality}$ / $N_{all}$',
                                             figname = folder + 'Prop_low_quality')

#[0.2345, 0.0035, 0.018]#[0.53, 0.87, 0.83]#[0.6, 1, 0.9]#[0.367, 0.833, 0.633]#
#e = [0.3065, 0.00546, 0.056]
#plt.ylabel('$N_{low quality samples}/N_{all}')
#plt.xlabel('Human queried')


if (to_same_plot is False) and save_figs:
    
    plt.gcf().savefig(figure_filenames[3] + '.pdf', transparent = True)
    plt.gcf().savefig(figure_filenames[3] + '.svg', transparent = True)
    plt.gcf().savefig(figure_filenames[3] + '.png', dpi=300)
