#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:23 2022

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
from hper_bo_simplified import bo_sim_target, acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
from scipy.special import erf, erfinv

import scipy as sp

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import tqdm
import time

import GPyOpt
import GPy

import psutil

#import logging

from functools import partial

# Reduces memory leakage with Spyder IDE. Otherwise not necessary.
import matplotlib
matplotlib.interactive(False)

def load_GP_model(path_model):
    """
    Load an already existing GP regression model as 
    GPy.models.gp_regression.GPRegression 

    Parameters
    ----------
    path_model : str
        Path to the pickle file that contains the GP regression model.

    Raises
    ------
    Exception
        When the model has not been saved into the pickle as
        GPy.models.gp_regression.GPRegression or GPyOpt.models.gpmodel.GPModel.

    Returns
    -------
    GPy.models.gp_regression.GPRegression
        
    """
    with open(path_model, 'rb') as f:
        
        gpmodel = pickle.load(f)
    
    if type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
        
        # Turn into GPRegression.
        gpmodel = gpmodel.model
        
    if type(gpmodel) is not GPy.models.gp_regression.GPRegression:
        
        # The function can read only GPy and GPyOpt GP regression models.
        raise Exception('Not implemented!')
    
    return gpmodel

def ternary_rand():

    # Initialization.
    x = 1
    y = 1

    # Enforce ternary proportions.
    i = 0
    while x + y > 1:
        [x, y] = np.random.rand(2)
        i = i+1  # np.random.seed((os.getpid() * int(time.time())) % 123456789)
        #y = np.random.rand()

    z = 1 - x - y

    return [x, y, z]


def ternary_rand_vector(n):

    v = []

    for i in range(n):
        v.append(ternary_rand())

    return v


def create_ternary_starting_points(n_reps=200, n_init=20):

    all_starting_points = []

    for i in range(n_reps):

        all_starting_points.append(ternary_rand_vector(n_init))

    return all_starting_points


def p_above(c_g, std=1):

    p = np.round(1 - erf(c_g/(std * np.sqrt(2))), 3)

    return p


def cg(p_above, std=1):

    c_g = np.round(np.sqrt(2) * std * erfinv(1-p_above), 3)

    return c_g


def build_filenames(folder, bo_params, acq_fun_descr, df_data_coll_descr, 
                    fetch_file_date=None, m=None):

    if fetch_file_date is None:

        # Create new files and folders.
        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())

        if m is not None:

            time_now = time_now + '-m' + str(m)

    else:

        # Existing files and folders.
        time_now = fetch_file_date

        if m is not None:

            time_now = time_now + '-m' + str(m)

    t_folder = folder + time_now + '/'

    if fetch_file_date is None:

        if not os.path.exists(t_folder):

            os.makedirs(t_folder)

    filename_prefix = t_folder + time_now

    filename_postfix = ('_nreps' + str(bo_params['n_repetitions']) +
                        '_nrounds' + str(bo_params['n_rounds']) +
                        '_inits' + str(bo_params['n_init']) +
                        '_batch' + str(bo_params['batch_size']) +
                        '_noisetarget' + str(bo_params['noise_target']) +
                        '_acq' + acq_fun_descr + df_data_coll_descr)

    #pickle_variable_names = ['optima', 'X_accum', 'Y_accum', 'data_fusion_data',
    #                         'BOmainresults', 'BO_lengthscales', 'BO_variances', 'BO_max_gradients']
    pickle_variable_names = ['optima', 'X_accum', 'Y_accum', 
                             'surrogate_model_params', 'data_fusion_params',
                             'BOobjects_suggs']
    
    pickle_filenames = []
    for i in pickle_variable_names:
        pickle_filenames.append(filename_prefix + '_' + i + filename_postfix)

    figs = ['_optimum', '_regretrA', '_Ndfpoints', # Used in this code
            '_region', '_lengthscales', '_variances', # Used in external scripts
            '_regretrB'
            ]

    figure_filenames = []
    for i in figs:
        figure_filenames.append(filename_prefix + i)

    return pickle_filenames, figure_filenames, t_folder


def modify_filename(filename, new_value, param_to_modify_str='_nreps'):

    # Has been tested only for nreps.

    idx0 = filename.find(param_to_modify_str) + len(param_to_modify_str)
    idx1 = idx0 + filename[idx0::].find('_')

    new_filename = filename[0:idx0] + str(new_value) + filename[idx1::]

    return new_filename

def set_bo_settings(bo_params, acquisition_function, jitter, 
                    data_fusion_property, df_data_coll_method, noise_df, 
                    c_grad, c_e):
    

    # Set acquisition function parameters (for this, you need to determine
    # if data fusion acquisition function is used or not.)
    
    if data_fusion_property is None:
        
        optional_data_fusion_settings = None
        
    else:
        
        optional_data_fusion_settings = {'df_target_property_name': data_fusion_property,
                                         'df_input_variables': bo_params['materials']
                                         }
        
    acq_fun_params = acq_param_builder(acquisition_function,
                                       optional_data_fusion_settings = optional_data_fusion_settings,
                                       #data_fusion_property=data_fusion_property,
                                       #data_fusion_input_variables=bo_params['materials'],
                                       #data_fusion_model = gt_model_human,
                                       optional_acq_settings = {'jitter': jitter}
                                       )
    
    acq_fun_descr = acq_fun_param2descr(
        acquisition_function, acq_fun_params=acq_fun_params)
    
    # Set data fusion data collection parameters.
    
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

    df_data_coll_descr = df_data_coll_method_param2descr(
        df_data_coll_params)
        
    return acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params

def repeated_tests(m, starting_point_candidates):#, gt_model_targetprop,
                   #gt_model_human):

    print(' ', end='', flush=True)
    
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, '\n')

    
    c_eig = [1, 0.8, 0.6, 0.4, 0.2, 0.1]#, 1]#[0.75]  # Expected information gain.
    # Size of the exclusion zone in percentage points (max. 100)
    c_exclz = [25, 20, 15, 10, 5, 2]#, 20]#[20]
    # Gradient limit. When the number is higher, the criterion picks less points. 0.05#, 0.07, 0.1, 0.2, 0.5, 0.75
    c_g = list(cg(np.array([0.9, 0.8, 0.6, 0.5, 0.2, 0.1])))#list(cg(np.array([0.6])))

    hyperparams_eig = []
    hyperparams_exclz = []
    for i in range(len(c_g)):
        for j in range(len(c_exclz)):

            hyperparams_exclz.append((c_g[i], c_exclz[j]))

        for j in range(len(c_eig)):

            hyperparams_eig.append((c_g[i], c_eig[j]))

    jitters = [0.1]

    n_eig = len(hyperparams_eig)
    n_exclz = len(hyperparams_exclz)
    n_hpars = 2 + n_eig + n_exclz
    n_j = len(jitters)

    folder = './Results/20240801/HO/Noise-free-jitter01/'
    ground_truth = [0.165, 0.04, 0.79] #[0.17, 0.03, 0.80]  # From C2a paper

    bo_params = {'n_repetitions': 25, # Repetitions of the whole BO process.
                 'n_rounds': 18, # Number of rounds in one BO.
                 'n_init': 3, # Number of initial sampling points.
                 'batch_size': 1, # Number of samples in each round.
                 'materials': ['CsPbI', 'MAPbI', 'FAPbI'], # Materials, i.e., search space variable names
                 'noise_target': 0  # Noise level of the target variable (between [0,1])
                 }
    
    noise_df = 0 # Noise level of the data fusion variable (between [0,1], used only if data fusion is used)

    # Give True if you don't want to run new BO but only fetch old results and re-plot them.
    fetch_old_results = False
    # Give False if you don't want to save the figures.
    save_figs = False
    # Give False if you don't want to save disk space while saving the data.
    save_disk_space = True
    # Give True if you want to close the figures immediately after they are created.
    close_figs = True
    
    log_progress = False
    
    #A I tried with seed = None.
    
    if (m > -1):

        if (m % n_hpars) == 0:

            data_fusion_property = None
            df_data_coll_method = None
            acquisition_function = 'EI'
            c_grad = None
            c_e = None
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[0]
            
        elif (m % n_hpars) == 1:

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_all'
            acquisition_function = 'EI_DFT'
            c_grad = None
            c_e = None
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[2]

        elif (m % n_hpars) < (n_hpars - n_exclz):

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_necessary_eig'
            c_grad = hyperparams_eig[(m % n_hpars)-2][0]
            c_e = hyperparams_eig[(m % n_hpars)-2][1]
            acquisition_function = 'EI_DFT'
            color = sn.color_palette()[1]
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            # color = np.array(sn.color_palette()[c_eig.index(c_e)+2])*(
            #    1 + c_g.index(c_grad) / len(c_g))
            # for i in range(len(color)):
            #    if color[i] > 1:
            #        color[i] = 1
            # =============================================================================

        else:

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_necessary_exclz'
            c_grad = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][0]
            c_e = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][1]
            acquisition_function = 'EI_DFT'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[3]

        jitter = jitters[m // n_hpars]
        
        ###############
        # Typically, one does not need to modify these inputs.
        
        acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params = set_bo_settings(
            bo_params, acquisition_function, jitter, data_fusion_property, 
            df_data_coll_method, noise_df, c_grad, c_e)
        
        # Create result folders and build filenames for result files.
        pickle_filenames, figure_filenames, triangle_folder = build_filenames(
            folder, bo_params, acq_fun_descr, df_data_coll_descr,
            fetch_file_date=fetch_file_date, m=m)
        
        #logging.basicConfig(filename= triangle_folder[0:-1] + '_log.txt', 
        #                    level=21, format='%(asctime)s - %(levelname)s - %(message)s')
        
        #if log_progress is False:
        #    
        #    logging.disable(logging.CRITICAL)
            
        #logging.log(31, "Starting method " + str(m))
                
        #logging.log(21, 'Jitter in reps: ' + str(acq_fun_params['jitter']))

        
        # Set figure style.
        mystyle = FigureDefaults('nature_comp_mat_sc')

        ###############

        all_starting_points = []
        bo_examples = []
        optima = []
        model_optima = []
        X_accum_all = []
        Y_accum_all = []
        data_fusion_params_all = []
        surrogate_model_params_all =  []
        #lengthscales_all = []
        #variances_all = []
        #max_gradients_all = []

        if fetch_old_results == False:

            for i in range(bo_params['n_repetitions']):

                all_starting_points.append(
                    starting_point_candidates[i][0:bo_params['n_init']])

                message = ('\n\nInit points method ' + str(m) + 
                             ',  repetition ' + str(i) + ':\n' +
                      str(all_starting_points[i]))
                #logging.info(message)

            for i in range(bo_params['n_repetitions']):

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
                    #bo_ground_truth_model_path = './Source_data/stability_model_GPyHomoscedastic', #'./Source_data/stability_model_improved_region_B', #
                    materials=bo_params['materials'],
                    rounds=bo_params['n_rounds'],
                    init_points=all_starting_points[i],
                    batch_size=bo_params['batch_size'],
                    acquisition_function=acquisition_function,
                    acq_fun_params=afp,
                    df_data_coll_params=ddcp,
                    no_plots=no_plots, results_folder=triangle_folder,
                    noise_target = bo_params['noise_target'],
                    seed = None, close_figs = close_figs)#Am)
                
                # Getting % usage of virtual_memory ( 3rd field)
                print('BO ended. \n')
                print('RAM memory % used:', psutil.virtual_memory()[2])
                
                
                optima.append(optimum)
                model_optima.append(model_optimum)
                
                X_accum_all.append(X_accum)
                Y_accum_all.append(Y_accum)

                if data_fusion_params is not None:
                    data_fusion_params_all.append(data_fusion_params)#['df_data_rounds'])
                
                surrogate_model_params_all.append(surrogate_model_params)
                #lengthscales_all.append(surrogate_model_params['lengthscales'])
                #variances_all.append(surrogate_model_params['variances'])
                #max_gradients_all.append(
                #    surrogate_model_params['max_gradients'])
                
                if ddcp is None:
                    
                    message = 'End method ' + str(m) + ': No data fusion, repetition ' + str(i)
                    
                else:
                    
                    message = 'End method ' + str(m) + ': ', ddcp, ', repetition ' + str(i)

                print(message)
                #logging.info(message)
                
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
                # also two times in between if the total number of repetitions
                # is large.
                if (i == (bo_params['n_repetitions']-1)) or (
                        (bo_params['n_repetitions'] > 15) and
                        (np.remainder((i+1),
                                      int(np.floor(bo_params['n_repetitions']/3)))
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
                

        else:

            # Fetch old results.

            pickle_variables = []

            # Fetch the results from pickled backup
            for s in pickle_filenames:
                dbfile = open(s, 'rb')
                pickle_variables.append(pickle.load(dbfile))
                dbfile.close()

            optima = pickle_variables[0]['optimal_samples']
            model_optima = pickle_variables[0]['model_optima']
            X_accum_all = pickle_variables[1]
            Y_accum_all = pickle_variables[2]
            surrogate_model_params_all = pickle_variables[3]
            data_fusion_params_all = pickle_variables[4]
            #results = pickle_variables[4]
            #lengthscales_all = pickle_variables[5]
            #variances_all = pickle_variables[6]
            #max_gradients_all = pickle_variables[7]

        # Does not work anymore:        
        #optima = np.array(optima)
        #model_optima = np.array(model_optima)/60
        
        n_rounds = bo_params['n_rounds'] #len(Y_accum_all[-1][-1])
        '''
        # Plot optimum vs number of BO samples collected.

        cols = ['Optimum' +
                x for x in list(map(str, range(n_rounds)))]
        df_optima_wide = pd.DataFrame(optima, columns=cols)  # range(n_rounds))
        df_optima_wide['Repeat'] = range(bo_params['n_repetitions'])
        df_optima_long = pd.wide_to_long(
            df_optima_wide, stubnames='Optimum', i='Repeat', j='Sample')

        plt.figure()
        sn.lineplot(data=df_optima_long, x='Sample',
                    y='Optimum', ci=90, color=color)
        plt.ylim([0, 10000])
        plt.tight_layout()

        if save_figs:

            plt.gcf().savefig(figure_filenames[0] + '.pdf', transparent=True)
            plt.gcf().savefig(figure_filenames[0] + '.svg', transparent=True)
            plt.gcf().savefig(figure_filenames[0] + '.png', dpi=300)

        plt.show()

        # Plot "X distance" as the regret vs BO rounds

        regrets = [[None for i in range(n_rounds)]
                   for i in range(bo_params['n_repetitions'])]

        for i in range(bo_params['n_repetitions']):

            Y_accum = Y_accum_all[i]
            X_accum = X_accum_all[i]

            for j in range(bo_params['n_rounds']):

                idx_optimum = np.argmin(Y_accum[j])
                X_optimum = X_accum[j][idx_optimum]
                regret = np.sqrt(np.sum((ground_truth - X_optimum)**2))
                regrets[i][j] = regret

            
        cols = ['Regret' +
                x for x in list(map(str, range(bo_params['n_rounds'])))]
        df_regrets_wide = pd.DataFrame(
            regrets, columns=cols)  # range(n_rounds))
        df_regrets_wide['Repeat'] = range(bo_params['n_repetitions'])
        df_regrets_long = pd.wide_to_long(
            df_regrets_wide, stubnames='Regret', i='Repeat', j='Round')

        plt.figure()
        sn.lineplot(data=df_regrets_long, x='Round',
                    y='Regret', ci=90, color=color)
        plt.ylim([0, 1])
        plt.tight_layout()

        if save_figs:

            plt.gcf().savefig(figure_filenames[1] + '.pdf', transparent=True)
            plt.gcf().savefig(figure_filenames[1] + '.svg', transparent=True)
            plt.gcf().savefig(figure_filenames[1] + '.png', dpi=300)

        plt.show()

        if acquisition_function == 'EI_DFT':

            # Plot N_data_fusion_data vs BO_rounds.

            n_df_points = []
            # Repeats
            for i in range(len(data_fusion_params_all)):

                n_df_points.append([])

                # BO rounds.
                for j in range(len(data_fusion_params_all[i])):

                    if j == 0:
                        n_df_points[i].append(
                            data_fusion_params_all[i][j].shape[0])

                    else:
                        n_df_points[i].append(
                            data_fusion_params_all[i][j].shape[0]+n_df_points[i][-1])

            cols = ['N_data_fusion_points' +
                    x for x in list(map(str, range(bo_params['n_rounds'])))]
            df_optima_wide = pd.DataFrame(
                np.array(n_df_points), columns=cols)  # range(n_rounds))
            df_optima_wide['Repeat'] = range(bo_params['n_repetitions'])
            df_optima_long = pd.wide_to_long(
                df_optima_wide, stubnames='N_data_fusion_points', i='Repeat', j='Round')

            plt.figure()
            sn.lineplot(data=df_optima_long, x='Round',
                        y='N_data_fusion_points', ci=90, color=color)
            # plt.ylim([0,450000/60])
            plt.tight_layout()

            if save_figs:

                plt.gcf().savefig(
                    figure_filenames[2] + '.pdf', transparent=True)
                plt.gcf().savefig(
                    figure_filenames[2] + '.svg', transparent=True)
                plt.gcf().savefig(figure_filenames[2] + '.png', dpi=300)

            plt.show()
            
            '''
    
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
    path_gtmodel_targetprop = './Source_data/stability_gt_model_GPR'#'./Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222' # GPy.models.gp_regression.GPRegression
    #path_gtmodel_targetprop = './Source_data/stability_model_equal_AB'
    path_gtmodel_humanevals = './Source_data/visualquality/human_model_scale0to1' #'./Source_data/visualquality/Human_GPR_model_20220801' # GPy.models.gp_regression.GPRegression
    
    # Number of methods to be tested.
    m_total = 74
    
    # Load the starting points for BO. Every method will
    # share these same init points.
    # shape: (repeat_idx, init_point_idx, composition)
    init_points = np.load('./Source_data/initpts.npy')
    
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
    
    # Number of cpus available to this job.
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    
    # Load source data models. The implemented code assumes these models do not
    # output scaled values but data in real units.
    global gt_model_targetprop
    gt_model_targetprop = load_GP_model(path_gtmodel_targetprop) # Perovskite stability data (units in [px*min]), 0 px*min is fully stable and high values are instable
    global gt_model_human
    gt_model_human = load_GP_model(path_gtmodel_humanevals) # Human opinion on sample quality data, scale [0,1], where 1 is bad quality and 0 is high quality.
    '''
    # This is a serial version of the code.
    for i in range(m_total):
        
        repeated_tests(i, starting_point_candidates = init_points)#,
                       #gt_model_targetprop = gt_model_targetprop,
                       #gt_model_human = gt_model_human)
    '''    
    # This is a parallelized version of the code.
    # Create a pool of workers (corresponding to Ncpus)
    with mp.Pool(ncpus) as pool:
        
        r = process_map(partial(repeated_tests,
                                starting_point_candidates=init_points),
                        range(m_total), max_workers=ncpus)
    
    
'''
Human does not look like it has been fitted properly. Check settings. Try to
fit a new GP with these hyperparam.
PyOpt.
'''