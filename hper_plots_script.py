#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:08:18 2024

@author: atiihone
"""
from hper_repetitions_simplified import build_filenames, cg, p_above, set_bo_settings
from hper_bo_simplified import predict_points, acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import seaborn as sn
import pandas as pd

# TO DO: Fix, it is not good to use both sn and figuredefaults, and figsize.
# from set_figure_defaults import FigureDefaults # C Antti Vepsalainen
# Set figure style.
#mystyle = FigureDefaults('nature_comp_mat_sc')

sn.set_context('paper')

cmap = mpl.cm.viridis
cmap.set_under(color='r')


def build_hyperparam_lists(c_eig, c_exclz, c_g, jitters):

    hyperparams_eig = []
    hyperparams_exclz = []
    for i in range(len(c_g)):
        for j in range(len(c_exclz)):

            hyperparams_exclz.append((c_g[i], c_exclz[j]))

        for j in range(len(c_eig)):

            hyperparams_eig.append((c_g[i], c_eig[j]))

    n_eig = len(hyperparams_eig)
    n_exclz = len(hyperparams_exclz)
    n_hpars = 2 + n_eig + n_exclz
    n_j = len(jitters)

    return hyperparams_eig, hyperparams_exclz, n_eig, n_exclz, n_hpars, n_j


def read_datetimes(folder):

    # List of subfolders within the folder.
    folder_list = np.sort(next(os.walk(folder), (None, [], None))[1])

    # Assumes that all the BO results folders start with digit, the rest are added
    # by the user and should not be considered.
    while folder_list[-1][0].isdigit() is False:

        folder_list = folder_list[0:-1]

    # Most functions in the BO repetitions code work based on datetime string
    # corresponding to each BO condition that is tested. We reuse the same
    # functions while plotting so let's fetch the datetime strings.
    dates_list = [folder_list[i][0:12] for i in range(len(folder_list))]

    return dates_list


def set_acqf_plotcolor_legend(m, n_hpars, n_exclz, hyperparams_eig,
                              hyperparams_exclz, jitters):

    if (m % n_hpars) == 0:

        # Vanilla BO
        data_fusion_property = None
        df_data_coll_method = None
        c_grad = None
        c_e = None
        acquisition_function = 'EI'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = None
        color = sn.color_palette()[0]
        name_in_legend = 'Vanilla'

    elif (m % n_hpars) == 1:

        # BO with human queried for every sample
        data_fusion_property = 'quality'
        df_data_coll_method = 'model_all'
        c_grad = None
        c_e = None
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = None
        color = sn.color_palette()[2]
        name_in_legend = 'Human always'

    elif (m % n_hpars) < (n_hpars - n_exclz):

        # BO with human queries based on expected information gain.
        data_fusion_property = 'quality'
        df_data_coll_method = 'model_necessary_eig'
        c_grad = hyperparams_eig[(m % n_hpars)-2][0]
        c_e = hyperparams_eig[(m % n_hpars)-2][1]
        acquisition_function = 'EI_DFT'
        color = sn.color_palette()[1]
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = None
        name_in_legend = 'EIG'

    else:

        # BO with human queries based on exclusion zones.
        data_fusion_property = 'quality'
        df_data_coll_method = 'model_necessary_exclz'
        c_grad = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][0]
        c_e = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][1]
        acquisition_function = 'EI_DFT'
        # Which data to fetch (if you only fetch and do not calculate new)?
        fetch_file_date = None
        color = sn.color_palette()[3]
        name_in_legend = 'Excl. zone'

    jitter = jitters[m // n_hpars]

    return acquisition_function, jitter, data_fusion_property, df_data_coll_method, c_grad, c_e, color, name_in_legend


def load_data(m, n_hpars, n_exclz, hyperparams_eig, hyperparams_exclz,
              jitters, bo_params, noise_df, folder, dates_list):

    acquisition_function, jitter, data_fusion_property, df_data_coll_method, c_grad, c_e, color, name_in_legend = set_acqf_plotcolor_legend(
        m, n_hpars, n_exclz, hyperparams_eig, hyperparams_exclz, jitters)

    fetch_file_date = dates_list[m]
    
    acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params = set_bo_settings(
        bo_params, acquisition_function, jitter, data_fusion_property, 
        df_data_coll_method, noise_df, c_grad, c_e)
    
    # Create result folders and build filenames for result files.
    pickle_filenames, figure_filenames, triangle_folder = build_filenames(
        folder, bo_params, acq_fun_descr, df_data_coll_descr,
        fetch_file_date=fetch_file_date, m=m)
    
    '''
    
    if df_property is None:
        
        optional_data_fusion_settings = None
        
    else:
        
        optional_data_fusion_settings = {'df_target_property_name': df_property,
                                         'df_input_variables': bo_params['materials']
                                         }
        
    acq_fun_params = acq_param_builder(af,
                                       optional_data_fusion_settings = optional_data_fusion_settings,
                                       #data_fusion_property=data_fusion_property,
                                       #data_fusion_input_variables=bo_params['materials'],
                                       #data_fusion_model = gt_model_human,
                                       optional_acq_settings = {'jitter': jitter}
                                       )
    
    acq_fun_descr = acq_fun_param2descr(
        af, acq_fun_params = acq_fun_params)
    
    ##
    #
    #acq_fun_params = acq_param_builder(af,
    #                                   optional_data_fusion_settings = optional_data_fusion_settings,
    #                                   #data_fusion_property=df_property,
    #                                   #data_fusion_input_variables=bo_params['materials'],
    #                                   optional_acq_settings={'jitter': jitter}
    #                                   )
    #acq_fun_descr = acq_fun_param2descr(af, acq_fun_params=acq_fun_params)

    if df_property is None:

        df_data_coll_params = df_data_coll_param_builder()

    elif (df_data_coll_method == 'model_all') or (df_data_coll_method == 'model_none'):

        df_data_coll_params = df_data_coll_param_builder(
            df_method=df_data_coll_method, noise_df=noise_df)

    else:

        df_data_coll_params = df_data_coll_param_builder(df_method=df_data_coll_method,
                                                         gradient_param=c_grad,
                                                         exclusion_param=c_e,
                                                         noise_df=noise_df)

    df_data_coll_descr = df_data_coll_method_param2descr(df_data_coll_params)

    pickle_filenames, figure_filenames, t_folder = build_filenames(folder, bo_params,
                                                                   acq_fun_descr,
                                                                   df_data_coll_descr,
                                                                   fetch_file_date=current_date,
                                                                   m=m)

    '''
    pickle_variables = []
    
    for s in pickle_filenames:
    
        # Fetch the results from pickled backup
        try:
            
            with open(s,'rb') as dbfile:
                
                pickle_variables.append(pickle.load(dbfile))
                dbfile.close()
                
        except IOError:
                
            pickle_variables.append(None)
            
            # BO objects variables are optional but the rest are not.
            if not 'BOobjects' in s:
                
                print('Loading this file failed: ', s)
            
    res_orig = {'optima': np.array(pickle_variables[0]['optimal_samples']),
                'model_optima': np.array(pickle_variables[0]['model_optima']),
                'X_accum': pickle_variables[1],
                'Y_accum': pickle_variables[2],
                'df_data': pickle_variables[4],
                'exemplary_run_data': pickle_variables[5],
                'surrogate_model_params': pickle_variables[3]
                }
    
    res_orig['lengthscales'] = []
    res_orig['variances'] = []
    res_orig['max_gradients'] = []
    
    for i in range(len(res_orig['surrogate_model_params'])):
        
        res_orig['lengthscales'].append(res_orig['surrogate_model_params'][i]['lengthscales'])
        res_orig['variances'].append(res_orig['surrogate_model_params'][i]['variances'])
        res_orig['max_gradients'].append(res_orig['surrogate_model_params'][i]['max_gradients'])
    
    # Save key results into more convenient Numpy arrays.

    n_repeats = len(res_orig['X_accum'])
    n_rounds = len(res_orig['X_accum'][0])
    n_dims = res_orig['X_accum'][0][0].shape[1]
    n_init = res_orig['X_accum'][0][0].shape[0]
    batch_size = res_orig['X_accum'][0][1].shape[0]-n_init
    n_samples = res_orig['X_accum'][0][-1].shape[0]

    X_accum = np.full((n_repeats, n_rounds, n_samples, n_dims), np.nan)
    Y_accum = np.full((n_repeats, n_rounds, n_samples, 1), np.nan)
    X_final = np.full((n_repeats, n_samples, n_dims), np.nan)
    Y_final = np.full((n_repeats, n_samples, 1), np.nan)
    X_opt = np.full((n_repeats, n_rounds, n_dims), np.nan)
    Y_opt = np.full((n_repeats, n_rounds, 1), np.nan)
    X_model_opt = np.full((n_repeats, n_rounds, n_dims), np.nan)
    Y_model_opt = np.full((n_repeats, n_rounds, 1), np.nan)
    
    
    if res_orig['df_data'] is not None:

        df_data_X_accum = np.full(
            (n_repeats, n_rounds, n_samples, n_dims), np.nan)
        df_data_Y_accum = np.full((n_repeats, n_rounds, n_samples, 1), np.nan)
        df_data_n_samples = np.full((n_repeats, n_rounds, 1), np.nan)

    else:

        df_data_X_accum = None
        df_data_Y_accum = None
        df_data_n_samples = None

    Y_opt = res_orig['optima'][:, :, [-1]]
    X_opt = res_orig['optima'][:, :, 0:-1]
    
    Y_model_opt = res_orig['model_optima'][:, :, [-1]]
    X_model_opt = res_orig['model_optima'][:, :, 0:-1]
    
    for i in range(n_repeats):

        # Variable to track the total number of data fusion data samples.
        n_df_samples = 0

        for j in range(n_rounds):

            # Datapoints accumulating by each round.
            n_samples_current = res_orig['X_accum'][i][j].shape[0]

            X_accum[i, j, 0:n_samples_current, :] = res_orig['X_accum'][i][j]
            Y_accum[i, j, 0:n_samples_current, :] = res_orig['Y_accum'][i][j]

            # Optimum value and location for each round.
            
            #idx_opt = np.argwhere(
            #    Y_accum[i, j, :, :] == res_orig['optima'][i, j])[0, 0]
            #Y_opt[i, j, :] = Y_accum[i, j, idx_opt, :]
            #X_opt[i, j, :] = X_accum[i, j, idx_opt, :]
            
            
            # Data fusion datapoints accumulating by each round.

            if len(res_orig['df_data']) != 0:

                # Data fusion data exists.

                #if j > 0:
                #
                #    # Copy data from previous round.
                #    df_data_X_accum[i, j, 0:n_df_samples, :] = df_data_X_accum[
                #        i, j-1, 0:n_df_samples, :]
                #    df_data_Y_accum[i, j, 0:n_df_samples, :] = df_data_Y_accum[
                #        i, j-1, 0:n_df_samples, :]
                
                n_df_samples_current = res_orig['df_data'][i]['df_data_accum'][j].shape[0]
                
                df_data_X_accum[i, j, 0:n_df_samples_current, :] = res_orig[
                    'df_data'][i]['df_data_accum'][j].iloc[:, 0:n_dims]
                df_data_X_accum[i, j, 0:n_df_samples_current, :] = res_orig[
                    'df_data'][i]['df_data_accum'][j].iloc[:, [-1]]

                #df_data_X_accum[i, j, n_df_samples:(n_df_samples +
                #                                    n_df_samples_current),
                #                :] = res_orig['df_data'][i][j].iloc[:, 0:n_dims]
                #df_data_Y_accum[i, j, n_df_samples:(n_df_samples +
                #                                    n_df_samples_current),
                #                :] = res_orig['df_data'][i][j].iloc[:, [-1]]

                n_df_samples = n_df_samples_current #n_df_samples + n_df_samples_current

                df_data_n_samples[i, j, :] = n_df_samples

        # Final dataset (target data).
        X_final[i, :, :] = res_orig['X_accum'][i][j]
        Y_final[i, :, :] = res_orig['Y_accum'][i][j]
    
    if res_orig['exemplary_run_data'] is not None:
        
        n_exemplary_runs = len(res_orig['exemplary_run_data'])
        
    else:
        
        n_exemplary_runs = 0

    bo_objects = []
    suggestions_for_next_X = np.full(
        (n_exemplary_runs, n_rounds, batch_size, n_dims), np.nan)

    for i in range(n_exemplary_runs):

        bo_objects.append(res_orig['exemplary_run_data'][i][0])

        #for j in range(n_rounds):
        #
        #    suggestions_for_next_X[i, j, :,
        #                           :] = res_orig['exemplary_run_data'][i][1][j].values

    res = {'optima': res_orig['optima'],
           'X_accum': X_accum,
           'Y_accum': Y_accum,
           'X_final': X_final,
           'Y_final': Y_final,
           'X_opt': X_opt,
           'Y_opt': Y_opt,
           'X_model_opt': X_model_opt,
           'Y_model_opt': Y_model_opt,
           'df_data_X_accum': df_data_X_accum,
           'df_data_Y_accum': df_data_Y_accum,
           'df_data_n_samples': df_data_n_samples,
           'bo_object_examples': bo_objects,
           'next_suggs_examples': suggestions_for_next_X,
           'lengthscales': res_orig['lengthscales'],
           'variances': res_orig['variances'],
           'max_gradients': res_orig['max_gradients'],
           'af': acquisition_function,
           'jitter': jitter,
           'df_property': data_fusion_property,
           'df_data_coll_method': df_data_coll_method
           }

    if df_data_coll_method is not None:

        if df_data_coll_method.find('exclz') != -1:

            res['c_exclz'] = c_e
            res['c_grad'] = c_grad

        elif df_data_coll_method.find('eig') != -1:

            res['c_eig'] = c_e
            res['c_grad'] = c_grad

    return res, res_orig, figure_filenames, color, name_in_legend


def test_outside_triangle(Xbest, plot_label = 'All'):

    n_dims = Xbest.shape[-1]
    n_repeats = Xbest.shape[0]

    if len(Xbest.shape) == 3:

        n_rounds = Xbest.shape[1]

    else:

        n_rounds = 1

    for i in range(n_repeats):

        s = np.zeros((n_rounds,))

        for j in range(n_rounds):

            if len(Xbest.shape) == 3:

                x_current = Xbest[i, j, :]

            else:

                x_current = Xbest[i, :]

            s[j] = np.sum(x_current, axis=0)

        ul = 1.005
        ll = 0.99

        if any(s > ul) or any(s < ll):

            print('Repeat ', i, 'has ', np.sum(s > ul) +
                  np.sum(s < ll), 'rounds outside when examining ' + plot_label + ' samples.')
            print(s, '\n')

            plt.figure()
            plt.plot(range(n_rounds), np.sum(
                Xbest[i, :, :], axis=1), 'k', linewidth=0.5)
            plt.plot(range(n_rounds), Xbest[i, :, :])
            plt.xlabel('Sample')
            plt.ylabel('$x_i$')
            plt.title(plot_label + ' samples on repeat ' + str(i))
            plt.legend(['Sum $x_i$', '$x_0$', '$x_1$', '$x_2$'])

            plt.show()


def calc_regret(ground_truth, X_opt, X_final, ground_truth_region_radius=0.1,
                regret_interm_idx=-1):

    # "X distance" as the regret vs BO rounds
    regret = np.sqrt(np.sum((ground_truth - X_opt)**2, axis=2))
    # Regret all samples
    regret_all_samples = np.sqrt(np.sum((ground_truth - X_final)**2,
                                        axis=2))

    # Number of samples in the total dataset within an acceptable distance from
    # the ground truth region.
    # Take only the requested first rounds of BO under consideration while
    # calculating this.
    # Note: X_final has typically more samples than X_opt
    # due to init_samples and possible batch size > 1.
    if regret_interm_idx != -1:

        # The number of points within the region are summed up only until
        # round with index regret_interm_idx.
        n_rounds_considered = regret_interm_idx + 1

        # The number of datapoints in the final dataset X_final may be higher
        # than in X_opt if n_init > 1 or batch size > 1.
        n_init = X_final.shape[1] - X_opt.shape[1] + 1
        batch_size = int((X_final.shape[1] - n_init)/(X_opt.shape[1] - 1))
        n_samples_considered = n_init + (n_rounds_considered - 1) * batch_size

    else:

        n_rounds_considered = X_opt.shape[1]
        n_samples_considered = X_final.shape[1]

    n_all_samples = np.sum(regret_all_samples[:, 0:n_samples_considered]
                           < ground_truth_region_radius,
                           axis=1)

    # Number of samples in the opt samples dataset within an acceptable
    # distance from the ground truth region.
    n_opt_samples = np.sum(regret[:, 0:n_rounds_considered]
                           < ground_truth_region_radius, axis=1)

    return regret, n_opt_samples, n_all_samples


def create_wide_and_long_df(values, var_name, n_rounds, n_repetitions):
    '''
    Format the data in the way that seaborn understands.
    '''

    if len(values.shape) > 2:

        values_temp = np.squeeze(values.copy())

    else:

        values_temp = values.copy()

    cols = [var_name + x for x in list(map(str, range(n_rounds)))]
    df_wide = pd.DataFrame(values_temp, columns=cols)
    df_wide['Repeat'] = range(n_repetitions)
    df_long = pd.wide_to_long(df_wide, stubnames=var_name, i='Repeat',
                              j='Round')

    return df_wide, df_long


def create_key_result_dataframes(dateslist, res_all, n_rounds, n_repetitions,
                                 opt_types, opt_key):

    optima_wides = []
    optima_longs = []
    regret_rA_wides = []
    regret_rA_longs = []
    regret_rB_wides = []
    regret_rB_longs = []
    n_df_wides = []
    n_df_longs = []
    n_samples_rA_wides = []
    n_samples_rA_longs = []
    n_samples_rB_wides = []
    n_samples_rB_longs = []

    for m in range(len(dateslist)):

        optima_wide, optima_long = create_wide_and_long_df(res_all[m]['Y_' + opt_types[opt_key]],
                                                           'Optimum $I_c$ (px$\cdot$h)',
                                                           n_rounds,
                                                           n_repetitions)

        regrets_rA_wide, regrets_rA_long = create_wide_and_long_df(res_all[m]['regret_rA'],
                                                                   'Regret',
                                                                   n_rounds,
                                                                   n_repetitions)

        regrets_rB_wide, regrets_rB_long = create_wide_and_long_df(res_all[m]['regret_rB'],
                                                                   'Regret$_{B}$',
                                                                   n_rounds,
                                                                   n_repetitions)

        optima_wides.append(optima_wide)
        optima_longs.append(optima_long)
        regret_rA_wides.append(regrets_rA_wide)
        regret_rA_longs.append(regrets_rA_long)
        regret_rB_wides.append(regrets_rB_wide)
        regret_rB_longs.append(regrets_rB_long)

        if res_all[m]['df_data_Y_accum'] is not None:

            n_df_wide, n_df_long = create_wide_and_long_df(res_all[m]['df_data_n_samples'],
                                                           'Human queries',
                                                           n_rounds,
                                                           n_repetitions)

            n_df_wides.append(n_df_wide)
            n_df_longs.append(n_df_long)

        else:

            n_df_wides.append(None)
            n_df_longs.append(None)

    res = [optima_wides, optima_longs, regret_rA_wides, regret_rA_longs,
           regret_rB_wides, regret_rB_longs, n_df_wides, n_df_longs]

    return res


def create_plot(df_data_long, name_in_legend, color, legend_list,
                figsize=[3, 2], fig_handle=None,
                ground_truth_reference=None, ylim=None, ylog=False,
                xlim=None):

    if fig_handle is None:
        current_fig = plt.figure()
    else:
        current_fig = fig_handle

    plt.figure(current_fig.number)
    ax = plt.axes()

    # Mean values
    sn.lineplot(data=df_data_long, x='Round', y=df_data_long.columns[0],
                ci=90, color=color, ax=ax)
    legend_list.append(name_in_legend)

    if (ylim is not None) and (ylog is False):
        plt.ylim([0, ylim])
    elif (ylim is not None) and (ylog is True):
        plt.ylim([1, ylim])

    if (xlim is not None):
        plt.xlim([0, xlim])

    # plt.xlim([0,50])

    if ground_truth_reference is not None:
        # Ground truth reference
        n_rounds = np.max(df_data_long.index.get_level_values('Round'))
        plt.plot([0, n_rounds], [ground_truth_reference, ground_truth_reference],
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
            legend_list = []  # Wipe for the next figure.

    # If fig_handle is not None: Multiple plots will be placed into the same fig,
    # Therefore legend are created only at the end of the loop.

    return current_fig, legend_list


def plot_check_ref_save(m, datatype_idx, dataframe_longs, gt_reference, fig_handle,
                        name_legend_all, colors_all, legend_list, ylim, ylog,
                        to_same_plot, save_figs, fig_names_all, xlim=None):

    if (to_same_plot is True) and (len(legend_list[datatype_idx]) > 0):

        # Do not plot ref line more than once into the same plot.
        gt_ref = None

    else:
        gt_ref = gt_reference

    current_fig, legend_list[datatype_idx] = create_plot(dataframe_longs[m], name_legend_all[m],
                                                         colors_all[m], legend_list[datatype_idx],
                                                         fig_handle=fig_handle,
                                                         ground_truth_reference=gt_ref,
                                                         ylim=ylim, ylog=ylog, xlim=xlim)

    if (to_same_plot is False) and (save_figs is True):

        plt.gcf().savefig(
            fig_names_all[m][datatype_idx] + '.pdf', transparent=True)
        plt.gcf().savefig(
            fig_names_all[m][datatype_idx] + '.svg', transparent=True)
        plt.gcf().savefig(fig_names_all[m][datatype_idx] + '.png', dpi=300)

    return current_fig, legend_list


def opt_plot(m, gt_y, optima_longs, fo, fig_names_all, to_same_plot, save_figs,
             legend_list, name_legend_all, colors_all):

    current_fig, legend_list = plot_check_ref_save(m, 0, optima_longs,
                                                   gt_y, fo, name_legend_all,
                                                   colors_all, legend_list,
                                                   None, False,
                                                   to_same_plot, save_figs,
                                                   fig_names_all)

    return current_fig, legend_list


def n_df_plot(m, gt_n, n_df_longs, fn, fig_names_all, to_same_plot, save_figs,
              legend_list, name_legend_all, colors_all):

    current_fig, legend_list = plot_check_ref_save(m, 2, n_df_longs,
                                                   gt_n, fn, name_legend_all,
                                                   colors_all, legend_list,
                                                   n_df_longs[m].index[-1][-1] + 1,
                                                   True,
                                                   to_same_plot, save_figs,
                                                   fig_names_all)

    return current_fig, legend_list


def regret_plot(m, gt_regret, regret_longs, fr, fig_names_all, to_same_plot, save_figs,
                legend_list, name_legend_all, colors_all, regret_interm_idx,
                datatype_idx, ylim=np.sqrt(2)):

    current_fig, legend_list = plot_check_ref_save(m, datatype_idx, regret_longs,
                                                   gt_regret, fr,
                                                   name_legend_all, colors_all, legend_list, ylim, False,
                                                   to_same_plot, save_figs, fig_names_all, xlim=regret_interm_idx)

    return current_fig, legend_list


def regret_ra_plot(m, gt_regret, regret_longs, fr, fig_names_all, to_same_plot, save_figs,
                   legend_list, name_legend_all, colors_all, regret_interm_idx,
                   ylim=1):  # np.sqrt(2)):

    current_fig, legend_list = regret_plot(m, gt_regret, regret_longs, fr,
                                           fig_names_all, to_same_plot,
                                           save_figs, legend_list,
                                           name_legend_all, colors_all,
                                           regret_interm_idx, datatype_idx=1,
                                           ylim=ylim)
    return current_fig, legend_list


def regret_rb_plot(m, gt_regret, regret_longs, fr, fig_names_all, to_same_plot, save_figs,
                   legend_list, name_legend_all, colors_all, regret_interm_idx,
                   ylim=np.sqrt(2)):

    current_fig, legend_list = regret_plot(m, gt_regret, regret_longs, fr,
                                           fig_names_all, to_same_plot,
                                           save_figs, legend_list,
                                           name_legend_all, colors_all,
                                           regret_interm_idx, datatype_idx=6,
                                           ylim=ylim)
    return current_fig, legend_list


def read_hpar_vectors(res_all):

    m_max = len(res_all)

    jitters = np.full((m_max,), np.nan)
    c_eigs = np.full((m_max,), np.nan)
    c_exclzs = np.full((m_max,), np.nan)
    c_grads = np.full((m_max,), np.nan)

    for m in range(m_max):

        if 'c_eig' in res_all[m]:

            c_eigs[m] = res_all[m]['c_eig']

        if 'c_exclz' in res_all[m]:

            c_exclzs[m] = res_all[m]['c_exclz']

        if 'c_grad' in res_all[m]:

            c_grads[m] = res_all[m]['c_grad']

        if 'jitter' in res_all[m]:

            jitters[m] = res_all[m]['jitter']

    # Reshape into (n_jitters, n_methods_per_jitter)
    n_jitters = len(np.unique(jitters))
    n_methods = m_max // n_jitters

    jitters = np.reshape(jitters, (n_jitters, n_methods))
    c_eigs = np.reshape(c_eigs, (n_jitters, n_methods))
    c_exclzs = np.reshape(c_exclzs, (n_jitters, n_methods))
    c_grads = np.reshape(c_grads, (n_jitters, n_methods))

    # Probabilities of being above the set gradient level are more relevant
    # for hyperparameter evaluation.
    p_aboves = p_above(c_grads)

    return c_eigs, c_exclzs, c_grads, jitters, p_aboves


def read_key_ho_results(res_all, n_df_wides, regret_rA_wides, regret_rB_wides,
                        n_jitters, n_methods, regret_interm_idx):

    m_max = len(res_all)

    n_humans = np.zeros((bo_params['n_rounds'], m_max))
    regrets_rA = np.full((bo_params['n_rounds'], m_max), 10**5)
    regrets_rB = np.full((bo_params['n_rounds'], m_max), 10**5)

    for m in range(m_max):

        if n_df_wides[m] is not None:

            # Last value is mean(repeats) so it is dropped.
            n_humans[:, m] = n_df_wides[m].mean().iloc[0:-1].values

        regrets_rA[:, m] = regret_rA_wides[m].mean().iloc[0:-1].values
        regrets_rB[:, m] = regret_rB_wides[m].mean().iloc[0:-1].values

    n_humans = n_humans[regret_interm_idx, :]
    regrets_rA = regrets_rA[regret_interm_idx, :]
    regrets_rB = regrets_rB[regret_interm_idx, :]

    n_humans = np.reshape(n_humans, (n_jitters, n_methods))
    regrets_rA = np.reshape(regrets_rA, (n_jitters, n_methods))
    #regrets_rB = np.reshape(regrets_rB, (n_jitters, n_methods))

    return n_humans, regrets_rA, regrets_rB


def plot_ho(res_all, bo_params, n_df_wides, regret_rA_wides, regret_rB_wides,
            regret_interm_idx, folder,  # n_humans_mean, n_eig,
            # n_exclz,
            plot_proportions=True, share_range=False, save_figs=True):

    # Rows are unique jitter values, columns are conditions for each jitter value.
    c_eigs, c_exclzs, c_grads, jitters, p_aboves = read_hpar_vectors(res_all)

    n_jitters = jitters.shape[0]
    n_methods = jitters.shape[1]
    n_humans, regrets_rA, regrets_rB = read_key_ho_results(res_all, n_df_wides, regret_rA_wides, regret_rB_wides,
                                                           n_jitters, n_methods, regret_interm_idx)

    if plot_proportions is True:

        # Number of human samples are determined by param n_init in the first round
        # of bo. After that, max 1 human sample is collected in each round (if no
        # batch mode).
        n_human_max = (bo_params['n_init'] + (bo_params['n_rounds'] - 1) *
                       bo_params['batch_size'])

        n_humans = n_humans / n_human_max

    if share_range is True:

        cbar_ticks = [np.linspace(np.min(n_humans) + 0.01, n_human_max, 5),
                      np.linspace(np.min(regrets_rA) + 0.01, np.sqrt(2), 5)]
    else:

        cbar_ticks = [None, None]

    for i in range(n_jitters):

        x_data = p_aboves[i, :]
        x_label = '$P_{above\,limit}$'
        y_data = [c_eigs[i, :], c_exclzs[i, :]]
        y_label = ['c$_{eig}$', 'c$_{exclz}$']
        filename_y = ['eig_', 'exclz_']
        z_data = [n_humans[i, :], regrets_rA[i, :]]
        z_label = ['$\overline{N}_{human}/\overline{N}_{human,\,max}$',
                   'Regret$_{' + str(regret_interm_idx+1) + ' rounds}$']
        filename_z = ['nhumans_', 'regrets_']
        #cbar_format_regret = '%.2f'
        #cbar_format_human = '%.2f'

        for j in range(len(y_data)):

            for k in range(len(z_data)):

                title = z_label[k] + ' (' + y_label[j] + \
                    ' method, jitter ' + str(jitters[i, 0]) + ')'
                filename = folder + 'ho_' + \
                    filename_y[j] + filename_z[k] + \
                    'jitter_' + str(jitters[i, 0])

                plt.figure()
                plt.scatter(x_data, y_data[j], c=z_data[k])
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label[j])
                plt.colorbar(ticks=cbar_ticks[k], extend='min',
                             label=z_label[k])  # , format = cbar_format_human)

                if save_figs is True:

                    plt.gcf().savefig(filename + '.pdf', transparent=True)
                    plt.gcf().savefig(filename + '.svg', transparent=True)
                    plt.gcf().savefig(filename + '.png', dpi=300)

                plt.show()


def calc_samples_within_ra_rb(res_all):

    n_repeats = res_all[0]['regret_rA_n_opt'].shape[0]
    n_methods = len(res_all)

    n_samples_rB = np.zeros((n_methods, n_repeats))
    n_samples_rA = np.zeros((n_methods, n_repeats))
    n_samples_rB_of_opt = np.zeros((n_methods, n_repeats))
    n_samples_rA_of_opt = np.zeros((n_methods, n_repeats))

    for m in range(n_methods):

        n_samples_rB[m, :] = res_all[m]['regret_rB_n_all']
        n_samples_rA[m, :] = res_all[m]['regret_rA_n_all']
        n_samples_rB_of_opt[m, :] = res_all[m]['regret_rB_n_opt']
        n_samples_rA_of_opt[m, :] = res_all[m]['regret_rA_n_opt']

    return n_samples_rA, n_samples_rB, n_samples_rA_of_opt, n_samples_rB_of_opt


def create_barplot(data_mean, data_std, name_in_legend, color,
                   figsize=[3, 2], ylabel=None,  fig_handle=None,
                   ground_truth_reference=None, ylim=1.00, figname='fig'):

    if fig_handle is None:
        current_fig = plt.figure()
    else:
        current_fig = fig_handle

    plt.figure(current_fig.number)

    ax = plt.axes()

    #bar = series_boolean.astype(int).mean()

    x = name_in_legend.copy()
    for i in range(len(x)):
        x[i] = x[i].replace("Vanilla", "Not")
        x[i] = x[i].replace("Human a", "A")
        x[i] = x[i].replace(" ", "\n")

    y = data_mean
    e = data_std

    # Mean values
    sn.barplot(x, y, ax=ax)
    plt.xticks(rotation=20)

    plt.ylabel(ylabel)
    plt.xlabel('Human queried')
    # if (ylim is not None):
    #plt.title('Searches that converge into Region A')
    # plt.ylim([0,ylim])

    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()

    plt.gcf().savefig(figname + '.pdf', transparent=True)
    plt.gcf().savefig(figname + '.svg', transparent=True)
    plt.gcf().savefig(figname + '.png', dpi=300)

    return current_fig


def plot_samples_within_ra_rb(n_samples_rA, n_samples_rB, n_samples_rA_of_opt, n_samples_rB_of_opt, name_legend_all, colors_all, fig_handle, folder, legends_visible):

    plt.figure()
    current_fig = create_barplot(np.mean(n_samples_rB, axis=1),
                                 np.std(n_samples_rB, axis=1),
                                 name_legend_all,
                                 colors_all,
                                 fig_handle=fb,
                                 ylabel=r'$N_{low}$ $_{quality}$ / $N_{all}$',
                                 figname=folder + 'Prop_low_quality_all_samples')
    plt.show()

    plt.figure()
    current_fig = create_barplot(np.mean(n_samples_rA, axis=1),
                                 np.std(n_samples_rA, axis=1),
                                 name_legend_all,
                                 colors_all,
                                 fig_handle=None,
                                 ylabel=r'$N_{region}$ $_{A}$ / $N_{all}$',
                                 figname=folder + 'Prop_region_A_all_samples')
    plt.show()

    plt.figure()
    current_fig = create_barplot(np.mean(n_samples_rB_of_opt, axis=1),
                                 np.std(n_samples_rB_of_opt, axis=1),
                                 name_legend_all,
                                 colors_all,
                                 fig_handle=None,
                                 ylabel=r'$N_{low}$ $_{quality}$ / $N_{opt}$',
                                 figname=folder + 'Prop_low_quality_opt_samples')
    plt.show()

    plt.figure()
    current_fig = create_barplot(np.mean(n_samples_rA_of_opt, axis=1),
                                 np.std(n_samples_rA_of_opt, axis=1),
                                 name_legend_all,
                                 colors_all,
                                 fig_handle=None,
                                 ylabel=r'$N_{region}$ $_{A}$ / $N_{opt}$',
                                 figname=folder + 'Prop_opt_region_opt_samples')
    plt.show()

###############################################################################
# SETTINGS - BO REPETITIONS


# Folder of experiments to plot.
# 20240423/Noiseless-BbetterthanA-noiseeste-12-add-constr-ard-norm-Mat52kernel/'#'20231129-noisytarget-noisyhuman-ho-j001/'#'./Results/triton/20230823-noisytarget-noisyhuman-ho/'
folder = './Results/20240801/HO/Noise-free-jitter01/'

# Experiments with the listed hyperparam range will be plotted.

# [0, 0.2, 0.5, 0.8, 1]#[0, 0.5, 0.75, 0.9, 1, 2]  # Expected information gain.
c_eig = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
# Size of the exclusion zone in percentage points (max. 100)
c_exclz = [25, 20, 15, 10, 5, 2]  # [0,1,5,10,20]#[1, 5, 10, 20]
# Gradient limit. When the number is higher, the criterion picks less points.
# 0.2, 0.5, 0.8, 1])))#list(cg(np.array([0.2, 0.5, 0.6, 0.8, 1])))
c_g = list(cg(np.array([0.9, 0.8, 0.6, 0.5, 0.2, 0.1])))

jitters = [0.1]

bo_params = {'n_repetitions': 25,
             'n_rounds': 18,
             'n_init': 3,
             'batch_size': 1,
             'materials': ['CsPbI', 'MAPbI', 'FAPbI'],
             # 1 means 100% of noise accoring to the std of the predictions of the ground truth model.
             'noise_target': 0
             }

# 1 means 100% of noise accoring to the std of the predictions of the ground truth model.
noise_df = 0

###############################################################################

# SETTINGS - PLOTS

# './Source_data/stability_model_improved_region_B'#
bo_ground_truth_model_path = './Source_data/stability_gt_model_GPR'#'./Source_data/stability_gt_model_GPR' #'./Source_data/stability_model_GPyHomoscedastic'

# ground_truth_rA = np.array([0.17, 0.03, 0.80])  # From C2a paper
# From the ground truth model
ground_truth_rA = np.array([[0.165, 0.04, 0.79]])
gt_rA_ref_level = 0.1  # Radius of the ground truth optimum region - regret
# converged below this level is within the region A.

ground_truth_rB = np.array([[1, 0, 0]])  # Ground truth location of Region B
gt_rB_ref_level = 1.15  # Regret of region B should reach this value when the
# run has converged into region A (not B!).

# Give False if you don't want to save the figures.
save_figs = True
# Plot everything to the same plot? Boolean.
to_same_plot = True
# Set legends off if you have many experiment conditions and want to see the
# curves better.
legends_visible = False

# When plotting HO results, it may be beneficial to plot+print intermediate
# regret values if the searches have all converged. Insert the round you want
# to plot in this case.
regret_interm_idx = 6

# Plot w.r.t. to optimum defined as the surrogate model optimum ('model_opt') 
# or as the sample optimum ('opt').
opt_key = 0
opt_types = {0: 'model_opt', 1: 'opt'}

###############################################################################
# INITIALIZE PLOT HANDLES AND KEY VARIABLES

materials = bo_params['materials']
n_rounds = bo_params['n_rounds']
n_repetitions = bo_params['n_repetitions']

# List of items in the legend, is filled later on.
legend_list = [[], [], [], [], [], [], [], []]

if to_same_plot:

    fo = plt.figure()  # Optimum figure
    fra = plt.figure()  # Regret from region A figure
    fn = plt.figure()  # N data fusion points figure
    fb = plt.figure()  # Simple bar plot on regions A/B
    fregion = None  # Plot not implemented
    fls = None  # Plot not implemented
    fvar = None  # Plot not implemented
    frb = plt.figure()  # Regret from region B figure

else:

    fo = None
    fra = None
    fn = None
    fb = None
    fregion = None  # Plot not implemented
    fls = None  # Plot not implemented
    fvar = None  # Plot not implemented
    frb = None

# Convenience variable that will be used for saving figures.
fig_handles = [fo, fra, fn, fb, fregion, fls, fvar, frb]

# Fetch ground truth model.
with open(bo_ground_truth_model_path, 'rb') as f:
    gt_stability_model = pickle.load(f)
# Fetch optimum y value from the ground truth model.
gt_y, _ = predict_points(gt_stability_model,
                                ground_truth_rA)#,
gt_y = gt_y[0,0] #np.ravel(gt_y)
#                                Y_data=gt_stability_model.Y)
#gt_y = 0#gt_y[0, 0]/60  # To px*h, from np array to float
#gt_y_var = gt_y_var[0, 0]/(60**2)

###############################################################################
# LOAD DATA

hyperparams_eig, hyperparams_exclz, n_eig, n_exclz, n_hpars, n_j = build_hyperparam_lists(
    c_eig, c_exclz, c_g, jitters)

dateslist = read_datetimes(folder)

res_all = []
fig_names_all = []
colors_all = []
name_legend_all = []

for m in range(len(dateslist)):

    print('Loading data for method', m, '...')
    res, results_orig, fig_names, color, name_legend = load_data(m, n_hpars, n_exclz,
                                                                 hyperparams_eig,
                                                                 hyperparams_exclz, jitters,
                                                                 bo_params, noise_df, folder,
                                                                 dateslist)
    # Results dictionary keys:
    # 'X_accum', 'Y_accum', 'X_final', 'Y_final', 'X_opt', 'Y_opt',
    # 'df_data_X_accum', 'df_data_Y_accum', 'df_data_n_samples',
    # 'bo_object_examples', 'next_suggs_examples', 'lengthscales', 'variances',
    # 'max_gradients', 'af', 'jitter', 'df_property', 'df_data_coll_method',
    # 'c_grad', 'c_e'

    res['Y_accum'] = res['Y_accum']#/60  # Units to px*h
    res['Y_final'] = res['Y_final']#/60
    res['Y_opt'] = res['Y_opt']#/60
    res['Y_model_opt'] = res['Y_model_opt']#/60
    
    # Confirm that all the sampled datapoints are inside the triangle. Will
    # print notifications and create figures if not.
    test_outside_triangle(res['X_final'], plot_label = 'All')
    test_outside_triangle(res['X_opt'], plot_label = 'Best')
    test_outside_triangle(res['X_model_opt'], plot_label = 'Model optimum')
    
    # Calculate X regrets w.r.t. ground truth optimum rA and the false
    # optimum rB.
    res['regret_rA'], res['regret_rA_n_opt'], res['regret_rA_n_all'] = calc_regret(
        ground_truth_rA, res['X_' + opt_types[opt_key]], res['X_final'], 
        gt_rA_ref_level, regret_interm_idx)

    res['regret_rB'], res['regret_rB_n_opt'], res['regret_rB_n_all'] = calc_regret(
        ground_truth_rB, res['X_' + opt_types[opt_key]], res['X_final'], 
        gt_rA_ref_level, regret_interm_idx)

    res_all.append(res)
    fig_names_all.append(fig_names)
    colors_all.append(color)
    name_legend_all.append(name_legend)

###############################################################################
# CREATE SEABORN-COMPATIBLE AND HUMAN-READABLE DATAFRAMES

[optima_wides, optima_longs, regret_rA_wides, regret_rA_longs, regret_rB_wides,
 regret_rB_longs, n_df_wides, n_df_longs] = create_key_result_dataframes(
    dateslist, res_all, n_rounds, n_repetitions, opt_types, opt_key)

###############################################################################
# PLOT CONVERGENCE

for m in range(len(dateslist)):

    # Optimum y sampled.
    current_fig, legend_list = opt_plot(m, gt_y, optima_longs,
                                        fo, fig_names_all, to_same_plot,
                                        save_figs, legend_list,
                                        name_legend_all, colors_all)

    # Regret region A (from optimum y sampled).
    current_fig, legend_list = regret_ra_plot(m, gt_rA_ref_level, regret_rA_longs,
                                              fra, fig_names_all, to_same_plot,
                                              save_figs, legend_list,
                                              name_legend_all, colors_all,
                                              regret_interm_idx)
    # Number of data fusion datapoints.

    if n_df_longs[m] is not None:

        current_fig, legend_list = n_df_plot(m, None, n_df_longs,
                                             fn, fig_names_all, to_same_plot,
                                             save_figs, legend_list,
                                             name_legend_all, colors_all)

    # Regret region B (from optimum y sampled).
    current_fig, legend_list = regret_rb_plot(m, gt_rB_ref_level, regret_rB_longs,
                                              frb, fig_names_all, to_same_plot,
                                              save_figs, legend_list,
                                              name_legend_all, colors_all,
                                              regret_interm_idx)

# Save plots if not done before.

if to_same_plot:

    plotname_postfixes = ['_' + opt_types[opt_key] + 'imum', '_regretrA', 
                          '_N_df_human', '', '', '', '', '_regretrB']
    
    plotname_prefix = 'All'

    for k in range(len(fig_handles)):

        current_fig = fig_handles[k]

        # If the figure has been created.
        if current_fig is not None:

            plt.figure(current_fig.number)

            if legends_visible:

                legend_list[k].append('Std')
                plt.legend(legend_list[k])

            if k == 2:  # Number of human queries

                for i in current_fig.axes:
                    i.set(yscale="log")

            if save_figs:

                plt.gcf().savefig(
                    folder + plotname_prefix + plotname_postfixes[k] + '.pdf', 
                    transparent=True)
                plt.gcf().savefig(
                    folder + plotname_prefix + plotname_postfixes[k] + '.svg', 
                    transparent=True)
                plt.gcf().savefig(
                    folder + plotname_prefix + plotname_postfixes[k] + '.png', 
                    dpi=300)

####################
# Show plots
plt.show()

plot_ho(res_all, bo_params, n_df_wides, regret_rA_wides, regret_rB_wides,
        regret_interm_idx, folder,
        #regrets_interm_mean, n_humans_mean, n_eig,
        # n_exclz,
        plot_proportions=True, share_range=False)


n_samples_rA, n_samples_rB, n_samples_rA_of_opt, n_samples_rB_of_opt = calc_samples_within_ra_rb(
    res_all)
n_samples = bo_params['n_init'] + \
    (bo_params['n_rounds']-1)*bo_params['batch_size']
plot_samples_within_ra_rb(n_samples_rA/n_samples, n_samples_rB/n_samples,
                          n_samples_rA_of_opt/n_samples,
                          n_samples_rB_of_opt/n_samples, name_legend_all,
                          colors_all, fb, folder, legends_visible)
