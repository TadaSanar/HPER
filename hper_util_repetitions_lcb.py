#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:46:10 2024

@author: atiihone
"""
import os
import datetime
import numpy as np
from hper_util_bo import acq_param_builder, acq_fun_param2descr, df_data_coll_param_builder, df_data_coll_method_param2descr
from scipy.special import erf, erfinv

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

def set_repeat_settings(m, c_g, c_exclz, c_eig, jitters):
    
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
    #n_j = len(jitters)
    
    if (m > -1):

        if (m % n_hpars) == 0:

            data_fusion_property = None
            df_data_coll_method = None
            acquisition_function = 'LCB'
            c_grad = None
            c_e = None
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            
        elif (m % n_hpars) == 1:

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_all'
            acquisition_function = 'LCB_DF'
            c_grad = None
            c_e = None
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            
        elif (m % n_hpars) < (n_hpars - n_exclz):

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_necessary_eig'
            c_grad = hyperparams_eig[(m % n_hpars)-2][0]
            c_e = hyperparams_eig[(m % n_hpars)-2][1]
            acquisition_function = 'LCB_DF'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
            
        else:

            data_fusion_property = 'quality'
            df_data_coll_method = 'model_necessary_exclz'
            c_grad = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][0]
            c_e = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][1]
            acquisition_function = 'LCB_DF'
            # Which data to fetch (if you only fetch and do not calculate new)?
            fetch_file_date = None
        
        jitter = jitters[m // n_hpars]
        
    return data_fusion_property, df_data_coll_method, acquisition_function, c_grad, c_e, jitter, fetch_file_date
