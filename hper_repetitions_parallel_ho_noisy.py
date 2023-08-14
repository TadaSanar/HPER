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
from hper_bo import bo_sim_target, acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
from scipy.special import erf, erfinv

import scipy as sp

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import tqdm
import time

#import logging

from functools import partial

# %load_ext autoreload
# %autoreload 2


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


def build_filenames(folder, bo_params, acq_fun_descr, df_data_coll_descr, fetch_file_date=None, m=None):

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
                        '_acq' + acq_fun_descr + df_data_coll_descr)

    pickle_variable_names = ['optima', 'X_accum', 'Y_accum', 'data_fusion_data',
                             'BOmainresults', 'BO_lengthscales', 'BO_variances', 'BO_max_gradients']

    pickle_filenames = []
    for i in pickle_variable_names:
        pickle_filenames.append(filename_prefix + '_' + i + filename_postfix)

    figs = ['_optimum', '_regret', '_Ndfpoints']

    figure_filenames = []
    for i in figs:
        figure_filenames.append(filename_prefix + i)

    return pickle_filenames, figure_filenames, t_folder


def modify_filename_nreps(filename, new_value, param_to_modify_str='_nreps'):

    # Has been tested only for nreps.

    idx0 = filename.find(param_to_modify_str) + len(param_to_modify_str)
    idx1 = filename[idx0::].find('_') + idx0

    new_filename = filename[0:idx0] + str(new_value) + filename[idx1::]

    return new_filename


def repeated_tests(m, starting_point_candidates):

    c_eig = [0, 0.5, 0.75, 0.9, 1, 2]  # Expected information gain.
    # Size of the exclusion zone in percentage points (max. 100)
    c_exclz = [1, 5, 10, 20]
    # Gradient limit. 0.05#, 0.07, 0.1, 0.2, 0.5, 0.75
    c_g = list(cg(np.array([0.2, 0.5, 0.6, 0.8, 1])))

    hyperparams_eig = []
    hyperparams_exclz = []
    for i in range(len(c_g)):
        for j in range(len(c_exclz)):

            hyperparams_exclz.append((c_g[i], c_exclz[j]))

        for j in range(len(c_eig)):

            hyperparams_eig.append((c_g[i], c_eig[j]))

    jitters = [0.01, 0.1]

    n_eig = len(hyperparams_eig)
    n_exclz = len(hyperparams_exclz)
    n_hpars = 2 + n_eig + n_exclz
    n_j = len(jitters)

    folder = './Results/20230811-noisytarget-noisyhuman-ho/'
    ground_truth = [0.17, 0.03, 0.80]  # From C2a paper

    bo_params = {'n_repetitions': 20,
                 'n_rounds': 75,
                 'n_init': 3,
                 'batch_size': 1,
                 'materials': ['CsPbI', 'MAPbI', 'FAPbI']
                 }

    # Give True if you don't want to run new BO but only fetch old results and re-plot them.
    fetch_old_results = False
    # Give False if you don't want to save the figures.
    save_figs = False
    # Choose if noisy queries are being used or exact.
    noise_df = True
    noise_target = True
    
    log_progress = False
    
    if (m > -1):

        if (m % n_hpars) == 0:

            data_fusion_property = None
            acquisition_function = 'EI'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[0]

        elif (m % n_hpars) == 1:

            data_fusion_property = 'visual'
            df_data_coll_method = 'model_all'
            acquisition_function = 'EI_DFT'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            color = sn.color_palette()[2]

        elif (m % n_hpars) < (n_hpars - n_exclz):

            data_fusion_property = 'visual'
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

            data_fusion_property = 'visual'
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

        acq_fun_params = acq_param_builder(acquisition_function,
                                           data_fusion_property=data_fusion_property,
                                           data_fusion_input_variables=bo_params['materials'],
                                           optional_acq_params={'jitter': jitter})
        acq_fun_descr = acq_fun_param2descr(
            acquisition_function, acq_fun_params=acq_fun_params)

        if data_fusion_property is None:

            df_data_coll_params = df_data_coll_param_builder()

        elif (df_data_coll_method == 'model_all') or (df_data_coll_method == 'model_none'):

            df_data_coll_params = df_data_coll_param_builder(
                df_method=df_data_coll_method)

        else:

            df_data_coll_params = df_data_coll_param_builder(df_method=df_data_coll_method,
                                                             gradient_param=c_grad,
                                                             exclusion_param=c_e)

        df_data_coll_descr = df_data_coll_method_param2descr(
            df_data_coll_params)

        pickle_filenames, figure_filenames, triangle_folder = build_filenames(
            folder, bo_params, acq_fun_descr, df_data_coll_descr,
            fetch_file_date=fetch_file_date, m=m)
        
        # Define log message file.
        # Level INFO corresponds to 21, and we don't want GPyOpt debug
        # messages on INFO level (there are way too many of them).
        '''
        # create logger
        logger = logging.getLogger('SpyderKernelApp')
        logger.setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(21)
        
        # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        
        
        # create file handler and set level to debug
        ch = logging.FileHandler(triangle_folder[0:-1] + '_log.txt')
        ch.setLevel(21)
        
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # add formatter to ch
        ch.setFormatter(formatter)
        
        # add ch to logger
        logger.addHandler(ch)
        
        logger.info("Starting")
                
        #logging.basicConfig(filename= triangle_folder[0:-1] + '_log.txt', 
        #                    level=21, format='%(asctime)s - %(levelname)s - %(message)s')
        
        if log_progress is False:
            
            logging.disable(logging.CRITICAL)
        '''
        # Set figure style.
        mystyle = FigureDefaults('nature_comp_mat_sc')

        ###############

        all_starting_points = []
        results = []
        optima = []
        X_accum_all = []
        Y_accum_all = []
        data_fusion_data_all = []
        lengthscales_all = []
        variances_all = []
        max_gradients_all = []

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

                else:

                    ddcp = df_data_coll_params.copy()

                next_suggestions, optimum, mod_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, bo_models = bo_sim_target(
                    bo_ground_truth_model_path='./Source_data/stability_model_GPyHomoscedastic',
                    materials=bo_params['materials'],
                    rounds=bo_params['n_rounds'],
                    init_points=all_starting_points[i],
                    batch_size=bo_params['batch_size'],
                    acquisition_function=acquisition_function,
                    acq_fun_params=afp,
                    df_data_coll_params=ddcp,
                    no_plots=no_plots, results_folder=triangle_folder,
                    noise_df=noise_df, noise_target=noise_target)

                # All ouputs saved only from the first two repetitions to save
                # disk space.
                if i < 2:
                    results.append([bo_models, next_suggestions])

                optima.append(optimum[:, -1])

                X_accum_all.append(X_accum)
                Y_accum_all.append(Y_accum)

                if data_fusion_params is not None:
                    data_fusion_data_all.append(
                        data_fusion_params['df_data_rounds'])

                lengthscales_all.append(surrogate_model_params['lengthscales'])
                variances_all.append(surrogate_model_params['variances'])
                max_gradients_all.append(
                    surrogate_model_params['max_gradients'])
                
                message = 'Method ' + str(m) + ', repetition ' + str(i)
                print(message)
                #logging.info(message)

                # Save results after all repetitions have been done but also two
                # times in between if the total number of repetitions is large.
                if (i == (bo_params['n_repetitions']-1)) or (
                        (bo_params['n_repetitions'] > 15) and
                        (np.remainder((i+1),
                                      int(np.floor(bo_params['n_repetitions']/3)))
                         == 0)):

                    pickle_variables = [optima, X_accum_all, Y_accum_all,
                                        data_fusion_data_all, results, lengthscales_all,
                                        variances_all, max_gradients_all]

                    # Save the results as an backup
                    for j in range(len(pickle_variables)):

                        if i < bo_params['n_repetitions']:

                            # Temporary filename for temp run safe-copies.
                            filename = modify_filename_nreps(
                                pickle_filenames[j], i+1)

                        else:

                            filename = pickle_filenames[j]

                        dbfile = open(filename, 'ab')
                        pickle.dump(pickle_variables[j], dbfile)
                        dbfile.close()

        else:

            # Fetch old results.

            pickle_variables = []

            # Fetch the results from pickled backup
            for s in pickle_filenames:
                dbfile = open(s, 'rb')
                pickle_variables.append(pickle.load(dbfile))
                dbfile.close()

            optima = pickle_variables[0]
            X_accum_all = pickle_variables[1]
            Y_accum_all = pickle_variables[2]
            data_fusion_data_all = pickle_variables[3]
            results = pickle_variables[4]
            lengthscales_all = pickle_variables[5]
            variances_all = pickle_variables[6]
            max_gradients_all = pickle_variables[7]

        optima = np.array(optima)/60

        # Plot optimum vs BO rounds.

        cols = ['Optimum' +
                x for x in list(map(str, range(bo_params['n_rounds'])))]
        df_optima_wide = pd.DataFrame(optima, columns=cols)  # range(n_rounds))
        df_optima_wide['Repeat'] = range(bo_params['n_repetitions'])
        df_optima_long = pd.wide_to_long(
            df_optima_wide, stubnames='Optimum', i='Repeat', j='Round')

        plt.figure()
        sn.lineplot(data=df_optima_long, x='Round',
                    y='Optimum', ci=90, color=color)
        plt.ylim([0, 10000])
        plt.tight_layout()

        if save_figs:

            plt.gcf().savefig(figure_filenames[0] + '.pdf', transparent=True)
            plt.gcf().savefig(figure_filenames[0] + '.svg', transparent=True)
            plt.gcf().savefig(figure_filenames[0] + '.png', dpi=300)

        plt.show()

        # Plot "X distance" as the regret vs BO rounds

        regrets = [[None for i in range(bo_params['n_rounds'])]
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
            for i in range(len(data_fusion_data_all)):

                n_df_points.append([])

                # BO rounds.
                for j in range(len(data_fusion_data_all[i])):

                    if j == 0:
                        n_df_points[i].append(
                            data_fusion_data_all[i][j].shape[0])

                    else:
                        n_df_points[i].append(
                            data_fusion_data_all[i][j].shape[0]+n_df_points[i][-1])

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

    del next_suggestions, optimum, mod_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, bo_models


if __name__ == "__main__":
    ###############################################################################

    m_total = 104

    # Create a list of seeds for repetitions (increase max_reps if you need
    # more repetitions than the current max_rep value is).
    max_reps = 200
    max_init_pts = 20
    starting_points = create_ternary_starting_points(
        n_reps=max_reps, n_init=max_init_pts)

    ###############################################################################
    # get number of cpus available to job
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()

    '''
    for i in range(m_total):
        
        repeated_tests(i, starting_point_candidates = starting_points)
        
    '''

    # create pool of ncpus workers
    with mp.Pool(ncpus) as pool:
        # apply work function in parallel

        # TO DO: Both versions work, which one is faster/better?
        #list(tqdm.tqdm(pool.imap(repeated_tests, range(m_total)), total=m_total))
        r = process_map(partial(repeated_tests,
                                starting_point_candidates=starting_points),
                        range(m_total), max_workers=ncpus)
