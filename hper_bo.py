"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import GPyOpt
import GPy
import pickle
import datetime
import matplotlib.pyplot as plt
from hper_plots_target import plotBO
from hper_plots_data_fusion import plotDF
#import logging
import random
import psutil

def predict_points(gpmodel, x_points, Y_data=None):
    '''
    For a GPy GP regression model or GPyOpt GPModel.
    '''
    if type(gpmodel) is GPy.models.gp_regression.GPRegression:
        
        # Prediction output is mean, variance.
        posterior_mean, posterior_var = gpmodel.predict_noiseless(x_points)
        posterior_std = np.sqrt(posterior_var)
        
    elif type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
        
        # Prediction output of the GPModel is mean, standard deviation. So let's
        # dig out the GPRegression model and predict with that.
        posterior_mean, posterior_var = gpmodel.model.predict_noiseless(x_points)
        #posterior_var = (posterior_std)**2
        posterior_std = np.sqrt(posterior_var)
        
    # If the model has been trained with already-scaled (zero mean, unit
    # variance) data, the provided train data 'Y_data' will be used for scaling
    # the predictions to the correct units.
    if Y_data is not None:
        posterior_mean_true_units = posterior_mean * \
            np.std(Y_data) + np.mean(Y_data)
        posterior_std_true_units = posterior_std * np.std(Y_data)

        posterior_mean = posterior_mean_true_units
        posterior_var = posterior_std_true_units**2
    
    return posterior_mean, posterior_var


def predict_points_noisy(gpmodel, x_points, Y_data=None, noise_level = 1,
                         seed = None):

    if seed is not None:
        np.random.seed(seed)
    # Predictions.
    posterior_mean, posterior_var = predict_points(
        gpmodel, x_points, Y_data=Y_data)
    
    if type(gpmodel) is GPy.models.gp_regression.GPRegression:
        
        gaussian_noise_variance = gpmodel.Gaussian_noise.variance
        
    elif type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
        
        gaussian_noise_variance = gpmodel.model.Gaussian_noise.variance
        
    if Y_data is not None:
        
        # Scale back.
        gaussian_noise_variance = gaussian_noise_variance * np.var(Y_data)
        
    # Adding Gaussian noise to the mean predictions.
    posterior_mean_noisy = np.random.normal(
        posterior_mean, np.sqrt(gaussian_noise_variance)*noise_level)#np.sqrt(posterior_var)*noise_level)
    
    #logging.log(21, 'Noise level: ' + str(noise_level))
    #logging.log(21, 'Posterior mean: ' + str(posterior_mean))
    #logging.log(21, 'Posterior mean noisy: ' + str(posterior_mean_noisy))
    #logging.log(21, 'Seed: ' + str(np.random.get_state()[1][0]))
    
    return posterior_mean_noisy, posterior_var, posterior_mean

def GP_model(data_fusion_data, data_fusion_target_variable = 'dGmix (ev/f.u.)', 
             lengthscale = 0.03, variance = 2, noise_variance = None,
             data_fusion_input_variables = ['CsPbI', 'MAPbI', 'FAPbI'],
             optimize_hyperpar = True, domain_boundaries = [0, 1]):
    
    if data_fusion_data is None:
        
        model = None
        
    else:
    
        if data_fusion_data.empty:
            
            model = None
            
        else:
            
            X = data_fusion_data[data_fusion_input_variables]
            Y = data_fusion_data[[data_fusion_target_variable]] 
            X = X.values # Optimization did not succeed without type conversion.
            Y = Y.values
            
            init_hyperpars, lims_kernel_var, lims_noise_var = evaluate_GP_model_constraints(
                Y, noise_variance, variance, lengthscale, 
                optimize_hyperpar = optimize_hyperpar, 
                domain_boundaries = domain_boundaries)
            
            # Define the kernel and model.
            
            kernel = GPy.kern.Matern52(input_dim=X.shape[1], 
                                  lengthscale=init_hyperpars['kernel_ls'], 
                                  variance=init_hyperpars['kernel_var'])
            
            model = GPy.models.GPRegression(X, Y, kernel, 
                                            noise_var = init_hyperpars[
                                                'noise_var'])
            
            constrain_optimize_GP_model(model, init_hyperpars = init_hyperpars,
                                        lims_kernel_var = lims_kernel_var,
                                        lims_noise_var = lims_noise_var, 
                                        optimize_hyperpar = optimize_hyperpar)
            
    return model

def evaluate_GP_model_constraints(Y, noise_variance, kernel_variance, 
                                    lengthscale, optimize_hyperpar = True,
                                    noise_var_limit = 1e-12, 
                                    domain_boundaries = [0, 1]):
    
    # Init value for noise_var, GPy will optimize it further.
    noise_var = noise_variance
    
    if ((optimize_hyperpar is True) and 
        ((noise_var is None) or (noise_var <= 0))):
        
        noise_var = 0.01*Y.var() # Initial assumption.
        
        # Noise_variance should not be zero for numerical stability.
        if noise_var == 0:
            
            noise_var = noise_var_limit
        
    #message = ('Human Gaussian noise variance in data and model input: ' +
    #           str(Y.var()) + ', ' + str(noise_var) + '\n' +
    #           'Human model data:' + str(Y))
    #print(message)
    #logging.log(21, message)
    
    # Hyperparameter initial guesses.
    
    kernel_var = kernel_variance
    
    if ((optimize_hyperpar is True) and 
        ((kernel_var is None) or (kernel_var <= 0))):
        
        kernel_var = Y.var() # Initial assumption
        
        # Kernel variance should not be zero (would allow only constant
        # values)
        if kernel_var == 0:
            
            kernel_var = 1
        
    kernel_ls = lengthscale
    
    if ((optimize_hyperpar is True) and 
        ((kernel_ls is None) or (kernel_ls <= 0))):
        
        kernel_ls = (domain_boundaries[1] - domain_boundaries[0])/2
        
        # Kernel lengthscale should not be zero for numerical stability.
        if kernel_ls == 0:
            
            kernel_ls = 1
    
    if optimize_hyperpar is True:
        
        noise_var_lower_limit = noise_var_limit
        
        # The upper bound is set to the noise level that corresponds to
        # the maximum Y value in the dataset.
        noise_var_upper_limit = noise_var + (Y.max())**2
            
        # Also kernel variance is limited because the model sometimes converged 
        # into ridiculous kernel variance values with small number of 
        # datapoints and no bounds on variance.
        kernel_var_lower_limit = kernel_var * noise_var_limit
        kernel_var_upper_limit = kernel_var + 10*(Y.max())**2
        
    else:
        
        noise_var_lower_limit = None
        noise_var_upper_limit = None
        kernel_var_lower_limit = None
        kernel_var_upper_limit = None
    
    init_hyperpars = {'noise_var': noise_var, 'kernel_var': kernel_var, 
                      'kernel_ls': kernel_ls}
    
    lims_kernel_var = [kernel_var_lower_limit, kernel_var_upper_limit]
    lims_noise_var = [noise_var_lower_limit, noise_var_upper_limit]
    
    return init_hyperpars, lims_kernel_var, lims_noise_var 
     
def constrain_optimize_GP_model(model, init_hyperpars = {'noise_var': None,
                                                         'kernel_var': None,
                                                         'kernel_ls': None},
                                lims_kernel_var = [None, None], 
                                lims_noise_var = [None, None], 
                                optimize_hyperpar = True, warning = False, 
                                verbose = False, max_iters = 1000, 
                                num_restarts = 2):
    
            if optimize_hyperpar is True: 
                
                
                # The upper bound is set to the noise level that corresponds to
                # the maximum Y value in the dataset.
                model.Gaussian_noise.constrain_bounded(lims_noise_var[0], 
                                                       lims_noise_var[1], 
                                                       warning = warning)
                
                # With small number of datapoints and no bounds on variance,
                # the model sometimes converged into ridiculous kernel variance
                # values.
                model.Mat52.variance.constrain_bounded(lims_kernel_var[0], 
                                                       lims_kernel_var[1],
                                                       warning = warning)
                
            else:
                
                # The upper bound is set to the noise level that corresponds to
                # the maximum Y value in the dataset.
                model.Gaussian_noise.constrain_fixed(init_hyperpars['noise_var'], 
                                                     warning = warning)
                
                # With small number of datapoints and no bounds on variance,
                # the model sometimes converged into ridiculous kernel variance
                # values.
                model.Mat52.variance.constrain_fixed(init_hyperpars['kernel_var'], 
                                                     warning = warning)
                
            # optimize
            model.optimize_restarts(max_iters = max_iters, 
                                    num_restarts = num_restarts, 
                                    verbose = verbose)
            
            #message = ('Human Gaussian noise variance in model output: ' + 
            #           str(model.Gaussian_noise.variance[0]))
            #logging.log(21, message)


def data_fusion_with_ei_df_param_builder(acquisition_function, 
                                          data_fusion_settings = None
                                          #data_fusion_target_property='dft',
                                          #data_fusion_input_variables=[
                             #'CsPbI', 'MAPbI', 'FAPbI'],
                         #optional_acq_params=None
                         ):
    
    '''
    This function builds a properly formatted param dictionary for EI_DF
    acquisition function when using bo_sim_target().

    The only acquisition_type implemented is 'EI_DF'. The allowed options for
    data_fusion_target_variable are 'dft', 'visual', or 'cutoff'. If the rest
    of the variables are None, hard-coded default values listed in this function
    will be resumed.
    
    '''

    if (acquisition_function != 'EI_DF'):
        
        raise Exception("This function has not been designed for the " +
                        "requested acquisition function: " + 
                        acquisition_function + ".")
    
    if data_fusion_settings is None:
        
        raise Exception("Data fusion settings values are needed for setting " +
                        "up data fusion. Give data_fusion_settings.")
    else:        
        
        # Init all params to None.
                
        p = {'lengthscale': None,
             'variance': None,
             'p_beta': None,
             'p_midpoint': None,
             'df_target_variable': None, 
             'df_input_variables': None}
        
        # Check if user has provided values for any of these parameters.
        for key in p:
            
            if key in data_fusion_settings:
                
                p[key] = data_fusion_settings[key]
            
        #p = {'df_input_variables': data_fusion_input_variables}

        #for i in range(len(list_p)):
        #
        #    p[list_p[i]] = None
        
        #if optional_acq_params is not None:
        #
        #    for i in range(len(list_p)):
        #
        #        if list_p[i] in optional_acq_params:
        #
        #            p.update({list_p[i]: optional_acq_params[list_p[i]]})
        
        # Else, pick the default values for the keys for each data type.
        
        if data_fusion_settings['df_target_property_name'] == 'dft':
            
            #if p['files'] is None:
            #
            #    # These files contain DFT data that is integrated into the optimization loop as
            #    # a soft constraint (starting from the round it is first listed - if the vector
            #    # is shorter than the number of rounds, it is assumed that no data is being
            #    # added to data fusion in the rounds not listed).
            #    files = [['./phasestability/CsFA/fulldata/CsFA_T300_above.csv',
            #              './phasestability/FAMA/fulldata/FAMA_T300_above.csv',
            #              './phasestability/CsMA/fulldata/CsMA_T300_above.csv']
            #             ]
            
            if p['df_target_variable'] == None:
                variable = 'dGmix (ev/f.u.)'
            if p['lengthscale'] == None:  # For Gaussian process regression
                lengthscale = 0.03
            if p['variance'] == None:  # For GPR
                variance = 2
            if p['p_beta'] == None:  # For probability model
                p_beta = 0.025
            if p['p_midpoint'] == None:  # For P model
                p_midpoint = 0  # For P

        elif data_fusion_settings['df_target_property_name'] == 'quality':

            #if p['files'] is None:
            #
            #    # Visual quality of the samples as a constraint.
            #    files = [
            #        ['./Source_data/visualquality/visualquality_round_0-1.csv']]
            
            if p['df_target_variable'] == None:
                variable = 'Quality'
            if p['lengthscale'] == None:  # For GP regression
                lengthscale = None #0.1 # Set a value for the lengthscale if you have info on this, otherwise the model learns it.
            if p['variance'] == None:  # For GPR
                variance = 0.5 #0.1 # Assumes the quality data is roughly zero mean unit variance.
            if p['p_beta'] == None:  # For probability model
                p_beta = 0.08#0.1
            if p['p_midpoint'] == None:  # For P model
                p_midpoint = 0.66#0.5  # For P

        elif data_fusion_settings['df_target_property_name'] == 'cutoff':

            #if p['files'] is None:
            #
            #    # Arbitrary function (such as a direct cutoff) as a constraint.
            #    files = [['./arbitrary_function/cutoff.csv']]
            
            if p['df_target_variable'] == None:
                variable = 'Cutoff'

            if p['lengthscale'] == None:  # For Gaussian process regression
                lengthscale = 0.05
            if p['variance'] == None:  # For GPR
                variance = 0.05
            if p['p_beta'] == None:  # For probability model
                p_beta = 0.025
            if p['p_midpoint'] == None:  # For P model
                p_midpoint = 0  # For P

        else:
            
            raise Exception('Data fusion target variable ' + 
                            data_fusion_settings['df_target_property_name'] +
                            ' has not been implemented in the parameter ' +
                            'builder. Please provide another variable name or' +
                            'add yours into the builder.')

        #if p['df_input_variables'] == None:
        #    input_variable = data_fusion_settings['df_input_variables']
        
        #if (files != None):
        #
        #    # Retrieve the data.
        #    data_fusion_data = []
        #
        #    for i in range(len(files)):
        #
        #        data_fusion_data.append([])
        #
        #        for j in range(len(files[i])):
        #            data_fusion_data[i].append(pd.read_csv(files[i][j]))
        #
        #        data_fusion_data[i] = pd.concat(data_fusion_data[i])
                
        ei_df_params = {'df_data': None, #data_fusion_data,
                         'df_target_var': variable,
                         'df_target_prop': data_fusion_settings['df_target_property_name'],
                         'df_input_var': p['df_input_variables'],
                         'gp_lengthscale': lengthscale,
                         'gp_variance': variance,
                         'p_beta': p_beta,
                         'p_midpoint': p_midpoint
                         }
    
    return ei_df_params


def acq_param_builder(acquisition_function, optional_data_fusion_settings = None,
                      #data_fusion_property=None,
                      #data_fusion_input_variables=None, 
                      optional_acq_settings = None):
    """
    Build a parameter dictionary to describe the acquisition function.
    Implemented only for EI and EI_DF acquisition functions.

    Parameters
    ----------
    acquisition_function : str
        Name of acquisition function
    optional_data_fusion_settings : dict, optional
        Parameters related to the data fusion. Dta fusion is not performed if
        the variable is None. The default is None.
    optional_acq_settings : dict, optional
        Optional parameters to be forwarded to the acquisition function. The 
        default is None.

    Raises
    ------
    Exception
        Raised when data fusion is requested but the acquisition function is
        other than EI_DF.

    Returns
    -------
    acq_fun_params : dict
        A dictionary with all the necessary parameters for calculating
        acquisition function values.

    """
    
    if acquisition_function == 'EI_DF':
        
        # Do data fusion. Set the parameters required for data fusion.
        acq_fun_params = data_fusion_with_ei_df_param_builder(acquisition_function,
                                              data_fusion_settings = optional_data_fusion_settings
                                              #data_fusion_target_property=data_fusion_property,
                                              #data_fusion_input_variables=data_fusion_input_variables,
                                              #optional_acq_params=optional_acq_params
                                              )

    #elif optional_data_fusion_settings is None:
    #    
    #    raise Exception(
    #    
    #        'Data fusion has not been implemented for this acquisition function: ' + acquisition_function)
    
    else:
        
        acq_fun_params = {}
    
    # Set the rest of acquisition function parameters.
    
    if (acquisition_function == 'EI') or (acquisition_function == 'EI_DF'):
    
        if optional_acq_settings is not None:
            
            if 'jitter' in optional_acq_settings:
                
                acq_fun_params['jitter'] = optional_acq_settings['jitter']
                
            else:
                
                acq_fun_params['jitter'] = 0.01
                
    
    return acq_fun_params


def acq_fun_param2descr(acq_fun, acq_fun_params=None):

    output_str = acq_fun

    if acq_fun == 'EI_DF':

        ei_df_params = acq_fun_params
        output_str = (output_str + '-dftarget-' + ei_df_params['df_target_prop'] +
                      '-lengthscale-' + str(ei_df_params['gp_lengthscale']) +
                      '-variance-' + str(ei_df_params['gp_variance']) +
                      '-p_beta-' + str(ei_df_params['p_beta']) +
                      '-p_midpoint-' + str(ei_df_params['p_midpoint']))
        
    if acq_fun_params is not None:
        
        if 'jitter' in acq_fun_params:
            
            output_str = output_str + '-jitter-' + str(acq_fun_params['jitter'])
        
    return output_str


def df_data_coll_param_builder(df_method=None, gradient_param=None, 
                               exclusion_param=None, noise_df = 1):

    if df_method is None:

        # No data fusion is done.
        df_data_coll_params = None

    else:

        if df_method.find('model') != -1:
        
            use_model = True
            noise = noise_df # Simulated noise
        
        else:
        
            use_model = False
            noise = None

        df_data_coll_params = {'use_model': use_model,
                               'noise_df': noise}
        
        if df_method.find('_necessary') != -1:

            if df_method.find('_eig') != -1:

                method = 'eig'
                c_g = gradient_param
                c_e = exclusion_param
                use_and = False
                df_data_coll_params['c_eig'] = c_e

            elif df_method.find('_excl') != -1:

                method = 'exclz'
                c_g = gradient_param
                c_e = exclusion_param
                use_and = False
                df_data_coll_params['c_exclz'] = c_e

            else:

                # This is a deprecated version that just exists for compatibility
                # with old results.
                method = 'necessary_depr'

            df_data_coll_params['c_grad'] = c_g
            df_data_coll_params['use_and'] = use_and

        elif df_method.find('_all') != -1:

            method = 'all'

        elif df_method.find('_none') != -1:

            method = 'none'

        else:

            raise Exception(
                'Data fusion data collection has not been implemented for this case.')

        df_data_coll_params['method'] = method

    return df_data_coll_params


def df_data_coll_method_param2descr(df_data_coll_params):

    output_str = ''

    if df_data_coll_params is not None:

        if df_data_coll_params['use_model']:

            output_str = output_str + '_model'
            output_str = output_str + '_noisedf' + str(df_data_coll_params['noise_df'])

        else:

            output_str = output_str + '_live'

        if ((df_data_coll_params['method'] == 'all') or
                (df_data_coll_params['method'] == 'none')):

            output_str = output_str + '_' + df_data_coll_params['method']

        else:

            output_str = output_str + '_necessary'

            if df_data_coll_params['method'] != 'necessary_depr':

                output_str = (output_str + '_' + df_data_coll_params['method'] +
                              '-grad_lim-' + str(df_data_coll_params['c_grad']))

                if df_data_coll_params['method'] == 'eig':

                    output_str = (output_str + '-eig_lim-' +
                                  str(df_data_coll_params['c_eig']))

                elif df_data_coll_params['method'] == 'exclz':

                    output_str = (output_str + '-r-' +
                                  str(df_data_coll_params['c_exclz']))

        if len(output_str) == 0:

            raise Exception(
                'Description has not been implemented for this case.')

    return output_str


def build_constraint_str(x_variables = ['CsPbI', 'MAPbI', 'FAPbI'], 
                         sum_range = [0.995, 1], prefix='x[:,', postfix=']'):
    """
    Builds a constraint string for GPyOpt that limits the search to the input
    range that is between the specified limits. GPyOpt constraints are of the
    form: "constraint < 0"
    
    The function can be used for constraining the search to a ternary subspace
    in a 3D search space: Default values of the function.
    
    Another use case is defining the same constraint in a 2-dimensional
    search space (to be used with 2-dimensional BO):
        
        build_constraint_str(x_variables = ['CsPbI', 'MAPbI'], 
                                 sum_range = [0, 1])

    Parameters
    ----------
    x_variables: list[str]
        List of x variable (material) names included in the constraint.
    sum_range : list[float]
        List of length two where the first element 
    prefix : TYPE, optional
        DESCRIPTION. The default is 'x[:,'.
    postfix : TYPE, optional
        DESCRIPTION. The default is ']'.

    Returns
    -------
    c0 : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.

    """
    c1 = ''
    c0 = str(sum_range[0])

    for idx in range(len(x_variables)):
        c1 = c1 + prefix + str(idx) + postfix + ' + '
        c0 = c0 + ' - ' + prefix + str(idx) + postfix

    c1 = c1[0:-2] + '- ' + str(sum_range[1])

    return c0, c1


'''
def load_ground_truth(path_model):

    # Load already existing stability data as the "ground truth" of stability.
    with open(path_model, 'rb') as f:
        target_model_raw = pickle.load(f)
    
    # The current implementation assumes the model is GPy gp_regression.
    # Uncomment the following lines if you want to use GPyOpt GPModel, instead.
    
    # The saved model is GPyOpt GPModel that is a wrapper of GPy gp_regression.
    # GPyOpt is not maintained anymore, so for easier adaptation to other
    # packages, we use here GPy gp_regression model. Let's dig it out.
    #target_model = target_model_raw.model
    
    target_model = target_model_raw
    
    return target_model
'''

def query_data_fusion_data_from_model(k, data_fusion_XZ_rounds, 
                                      data_fusion_XZ_accum, init_points, 
                                      data_fusion_x_next, gt_model_datafusionprop, 
                                      rounds, materials, 
                                      acq_fun_params, noise_level = 1,
                                      seed = None):

    # Data fusion data is generated using the ground truth
    # model.

    if k == 0:
        
        # Sample the initial points given by the user.
        
        # Predict.
        
        # Note: There's no need to add train data to this GP model
        # because it has not been trained with scaled Y data (unlike
        # the target data model).

        if noise_level == 0:

            data_fusion_z_next = predict_points(
                gt_model_datafusionprop, np.array(init_points))[0]
        else:

            data_fusion_z_next = predict_points_noisy(
                gt_model_datafusionprop, np.array(init_points),
                noise_level = noise_level, seed = seed)[0]

        # Add to the list.
        data_fusion_XZ_rounds[k] = pd.DataFrame(
            np.concatenate((init_points,
                            data_fusion_z_next), axis=1),
            columns=acq_fun_params['df_input_var'] +
            [acq_fun_params['df_target_var']])
        
    else:

        # There may be point(s) suggested in the previous round
        # that need to be evaluated.

        if data_fusion_x_next[k-1].empty is False:

            # Note: There's no need to add train data to this GP model
            # (see above).

            # Predict.
            if noise_level == 0:

                data_fusion_z_next = predict_points(
                    gt_model_datafusionprop, data_fusion_x_next[k-1].values)[0]
            else:

                data_fusion_z_next = predict_points_noisy(
                    gt_model_datafusionprop, data_fusion_x_next[k-1].values,
                    noise_level = noise_level, seed = seed)[0]

            # Add to the list.
            data_fusion_XZ_rounds[k] = pd.DataFrame(
                np.concatenate((data_fusion_x_next[k-1].values,
                                data_fusion_z_next), axis=1),
                columns=acq_fun_params['df_input_var'] +
                [acq_fun_params['df_target_var']])
        else:

            data_fusion_XZ_rounds[k] = pd.DataFrame(
                columns=acq_fun_params['df_input_var'] +
                [acq_fun_params['df_target_var']])
            
    data_fusion_XZ_accum = fill_accum_df_with_this_round(
                data_fusion_XZ_accum, data_fusion_XZ_rounds, k)

    return data_fusion_XZ_rounds, data_fusion_XZ_accum


def fill_accum_df_with_this_round(accum_df, rounds_df, k):

    if k == 0:

        accum_df[k] = rounds_df[k].copy()

    else:

        accum_df[k] = pd.concat([accum_df[k-1], rounds_df[k]],
                                ignore_index=True)

    return accum_df


def fill_accum_ar_with_this_round(accum_ar, rounds_df, k):

    if k == 0:

        accum_ar[k] = rounds_df[k].values.copy()

    else:

        accum_ar[k] = np.append(accum_ar[k-1], rounds_df[k].values, axis=0)

    return accum_ar


def query_target_data_from_model(k, X_rounds, Y_rounds, X_accum, Y_accum,
                                 init_points, x_next, gt_model_targetprop, rounds,
                                 materials, noise_level = 1,
                                 seed = None):

    if (k == 0):

        # Initialization with the given grid points.
        X_rounds[k] = pd.DataFrame(init_points, columns=materials)
        
    else:

        # The locations suggested after the previous BO round will be
        # sampled in this round.
        X_rounds[k] = pd.DataFrame(x_next[k-1], columns=materials)

        #compositions_input.append(pd.DataFrame(x_next[k-1], columns = materials))

        #df = compositions_input[k].copy()

    if noise_level == 0:
        
        preds = predict_points(gt_model_targetprop, X_rounds[k].values)[0]
    else:
        
        preds = predict_points_noisy(gt_model_targetprop, X_rounds[k].values, 
                                     noise_level = noise_level,
                                     seed = seed)[0]
        
    # np.reshape(preds, (preds.shape[0], 1))
    Y_rounds[k] = pd.DataFrame(preds, columns=['Target value'])

    # Y_rounds[k] = np.reshape(predict_points_noisy(stability_model, x, stability_model.Y)[0],
    #                         (predict_points_noisy(stability_model, x, stability_model.Y)[0].shape[0], 1))

    #df['Merit'] = predict_points_noisy(stability_model, x, stability_model.Y)[0]
    # degradation_input.append(df)

    # X is a vector of compositions, Y is a vector of merit values.
    #X_rounds[k] = df[materials].values
    # Reshaping is done to go from (n,) to (n,1), which is required by GPyOpt.
    #Y_rounds[k] = np.reshape(df['Merit'].values, (df['Merit'].values.shape[0], 1))

    Y_accum = fill_accum_ar_with_this_round(Y_accum, Y_rounds, k)
    X_accum = fill_accum_ar_with_this_round(X_accum, X_rounds, k)

    # For each BayesianOpt round, we include only the data that has been
    # collected by that time.

    # for j in range(rounds):
    #
    #    if j >= k:
    #
    #        if k == 0:
    #
    #            X_accum[j] = X_rounds[k].values
    #            Y_accum[j] = Y_rounds[k].values
    #
    #        else:
    #
    #            X_accum[j] = np.append(X_accum[j], X_rounds[k].values, axis=0)
    #            Y_accum[j] = np.append(Y_accum[j], Y_rounds[k].values, axis=0)

    return X_rounds, Y_rounds, X_accum, Y_accum


def determine_data_fusion_points(data_fusion_XZ_accum,
                                 df_data_coll_params, acq_fun_params, x_next,
                                 current_surrogate_model_params, materials, bounds, k):

    # The locations from which the data fusion data for _the next round_ will
    # be queried are determined here.

    if df_data_coll_params['method'] == 'none':

        # 'model_none' or 'live_none'

        # Never sample from the model or query human, just add an empty df for
        # records.
        result = pd.DataFrame(columns=acq_fun_params['df_input_var'])

    elif df_data_coll_params['method'] == 'all':

        # Always sample human.
        result = pd.DataFrame(x_next[k],
                              columns=acq_fun_params['df_input_var'])

        if df_data_coll_params['use_model'] == False:

            # Request humans to give feedback from the specified samples.
            print('Give feedback on sample quality of these samples:\n',
                  result)

    elif ((df_data_coll_params['method'] == 'exclz') or
          (df_data_coll_params['method'] == 'eig')):

        # Gradient criterion.
        
        # Constant for the gradient limit.
        c_grad = df_data_coll_params['c_grad']

        # Parameters of the current surrogate model for the optimization target.
        lengthscale_s = current_surrogate_model_params['lengthscale']
        variance_s = current_surrogate_model_params['variance']

        # Sample only if the suggested point has a larger gradient than this
        # limit.
        gradient_limit = np.max((np.sqrt(variance_s)/lengthscale_s)*c_grad)

        # Max. gradients of the next suggested points in the surrogate model.
        grad_max_s_next = current_surrogate_model_params['max_gradient']

        # Pick new points for which the surrogate model has a high gradient, no
        # matter if there is an earlier data fusion point nearby.
        new_df_points_x_g = x_next[k][grad_max_s_next > gradient_limit]
        
        # THIS OPTION IS NOT IN USE.
        # Pick new points for which the surrogate model does not have
        # a high gradient but they are located far away from the
        # previously sampled points (in unknown region).
        new_df_points_x_u = x_next[k][grad_max_s_next <= gradient_limit]

        # If there are points to be considered based on exclusion zone or
        # gradient criterion.
        if new_df_points_x_g.shape[0] > 0:

            # Drop the points that are excluded from the points to be queried
            # because there are previous human evaluations nearby.
            if df_data_coll_params['method'] == 'exclz':

                # Drop points with an earlier data fusion point nearby.
                # 'Nearby' is X% of the domain length here.
                c_exclz = df_data_coll_params['c_exclz']
                r_limit = (bounds[0]['domain'][1] - bounds[0]
                           ['domain'][0])*c_exclz/100

                index = 0
                for l in range(len(new_df_points_x_g)):  # Should be u finally!

                    if data_fusion_XZ_accum[k].shape[0] > 0:

                        if np.any(np.sqrt(np.sum((
                                data_fusion_XZ_accum[k].iloc[:, 0:len(materials)] -
                                new_df_points_x_g[index])**2, axis=1)) <
                                r_limit):

                            new_df_points_x_g = np.delete(new_df_points_x_g,
                                                          index, axis=0)
                            # TO DO: Test if index works correctly when batch BO is used!
                            message = 'Deleted a point based on r exclusion.'
                            #logging.log(21, message)

                        else:

                            index = index + 1

            elif df_data_coll_params['method'] == 'eig':

                # If the data fusion model exists already.
                if data_fusion_XZ_accum[k].shape[0] > 0:

                    # Drop points if the expected information gain for the
                    # human opinion model is too low.

                    ## Let's re-create the human opinion model for EIG test.
                    #current_df_model = GP_model(data_fusion_XZ_accum[k],
                    #                            acq_fun_params['df_target_var'],
                    #                            acq_fun_params['gp_lengthscale'],
                    #                            acq_fun_params['gp_variance'],
                    #                            acq_fun_params['df_input_var'])
                    current_df_model = acq_fun_params['df_model']
                    
                    # Variance on each point x (pred. from the data fusion
                    # model).
                    var_d_next = predict_points(
                        current_df_model, new_df_points_x_g)[1]

                    # Data fusion model y variance estimate.
                    vary_d = current_df_model.Gaussian_noise.variance[0]
                    message = 'Data fusion Gaussian noise variance: ' + str(vary_d)
                    #logging.log(21, message)

                    index = 0
                    for l in range(len(new_df_points_x_g)):

                        eig = 0.5 * (np.log10(var_d_next[l, 0]/vary_d + 1))
                        c_eig = df_data_coll_params['c_eig']

                        # Let's scale EIG scale factor
                        # based on max EIG (i.e., c_eig = 1 samples very little,
                        # c_eig = 0 does not limit at all).
                        eig_max = - np.log10(1/2)

                        if eig < (eig_max * c_eig):

                            new_df_points_x_g = np.delete(
                                new_df_points_x_g, index, axis=0)
                            message = 'Deleted a point based on EIG.'
                            #logging.log(21, message)

                        else:

                            index = index + 1

        # NOT IN USE AT THE MOMENT.
        # Combine the two criteria.
        # np.append(new_df_points_x_g, new_df_points_x_u, axis = 0)
        new_df_points_x = new_df_points_x_g

        if new_df_points_x.shape[0] > 0:

            result = pd.DataFrame(new_df_points_x,
                                  columns=acq_fun_params['df_input_var'])

            if df_data_coll_params['use_model'] == False:

                # Request humans to give feedback from the specified samples.
                print('Give feedback on sample quality of these samples:\n',
                      result)
        else:

            # Add empty DataFrame
            result = pd.DataFrame(columns=acq_fun_params['df_input_var'])

    return result

def plot_basic(Y_accum, X_accum, optimum, model_optimum, rounds, time_str = None, results_folder = None, 
               ref_x = np.array([[0.165, 0.04, 0.79]]), ref_y = 126444, 
               ref_y_std = 106462, saveas = False, hyperpars = None,
               data_fusion_hyperpars = None, close_figs = True):
    
        plt.figure()
        plt.plot(range(Y_accum[-1].shape[0]), Y_accum[-1])
        plt.xlabel('Sample')
        plt.ylabel('Target value')
        plt.title('All samples')
        
        if results_folder is not None:
            
            filename = results_folder + 'All-y-' + time_str
        
        plt.tight_layout()
        
        if saveas:
            
            plt.gcf().savefig(filename + '.pdf', transparent = True)
            
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
        
        plt.figure()
        plt.plot(range(X_accum[-1].shape[0]), np.sum(X_accum[-1], axis = 1), 'k', linewidth = 0.5)
        plt.plot(range(X_accum[-1].shape[0]), X_accum[-1])
        plt.xlabel('Sample')
        plt.ylabel('$x_i$')
        plt.title('All samples')
        plt.legend(['Sum $x_i$', '$x_0$', '$x_1$', '$x_2$'])
                
        if results_folder is not None:
            
            filename = results_folder + 'All-x-' + time_str
            
        plt.tight_layout()
            
        if saveas:
                
            plt.gcf().savefig(filename + '.pdf', transparent = True)
        
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
        
        plt.figure()
        plt.plot(range(rounds), optimum[:, -1])
        plt.plot((0, rounds), [ref_y, ref_y], 'k--', linewidth = 0.5)
        plt.xlabel('Round')
        plt.ylabel('Target value')
        plt.title('Best found sample')
                
        if results_folder is not None:
            
            filename = results_folder + 'Best-sample-y' + time_str
        
        plt.tight_layout()
        
        if saveas:
            
            plt.gcf().savefig(filename + '.pdf', transparent = True)
            
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
                
        
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:ref_x.shape[1]]
        
        plt.figure()
        plt.plot(range(rounds), np.sum(optimum[:, 0:(-1)], axis = 1), 'k', linewidth = 0.5)
        plt.plot(range(rounds), optimum[:, 0:(-1)])
        
        for i in range(ref_x.shape[1]):
            
            plt.plot((0, rounds-1), np.repeat(ref_x[0,i], repeats = 2, axis = 0), 
                     '--', linewidth = 0.5, color = colors[i])
            
        plt.xlabel('Round')
        plt.ylabel('$x_i$')
        plt.title('Best found sample')
        plt.legend(['Sum $x_i$', '$x_0$', '$x_1$', '$x_2$'])
                
        if results_folder is not None:
            
            filename = results_folder + 'Best-sample-x' + time_str
        
        plt.tight_layout()
        
        if saveas:
            
            plt.gcf().savefig(filename + '.pdf', transparent = True)
            
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
        
        plt.figure()
        plt.plot(range(rounds), model_optimum[:, -1])
        plt.plot((0, rounds), [ref_y, ref_y], 'k--', linewidth = 0.5)
        plt.xlabel('Round')
        plt.ylabel('Target value')
        plt.title('Model optimum')
                
        if results_folder is not None:
            
            filename = results_folder + 'Model-opt-y' + time_str
        
        plt.tight_layout()
        
        if saveas:
            
            plt.gcf().savefig(filename + '.pdf', transparent = True)
            
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
                
        plt.figure()
        plt.plot(range(rounds), np.sum(model_optimum[:, 0:-1], axis = 1), 'k', 
                 linewidth = 0.5)
        plt.plot(range(rounds), model_optimum[:, 0:-1])
        
        for i in range(ref_x.shape[1]):
            
            plt.plot((0, rounds), np.repeat(ref_x[0,i], repeats = 2, axis = 0), 
                     '--', linewidth = 0.5, color = colors[i])
            
        plt.xlabel('Round')
        plt.ylabel('$x_i$')
        plt.title('Model optimum')
        plt.legend(['Sum $x_i$', '$x_0$', '$x_1$', '$x_2$'])
                
        if results_folder is not None:
            
            filename = results_folder + 'Model_opt-x' + time_str
        
        plt.tight_layout()
        
        if saveas:
            
            plt.gcf().savefig(filename + '.pdf', transparent = True)
            
        if close_figs:
            
            plt.close()
            
        else:
            
            plt.show()
        
        if hyperpars is not None:
            
            for i in hyperpars.keys():
                
                plt.figure()
                plt.plot(range(rounds), hyperpars[i])
                
                plt.xlabel('Round')
                plt.ylabel('Hyperparameter ' + i)
                
                if 'lengthscale' in i:
                    
                    plt.ylim((0,2))
                        
                if results_folder is not None:
                    
                    filename = results_folder + 'Hyperpar-' + i
                
                plt.tight_layout()
                
                if saveas:
                    
                    plt.gcf().savefig(filename + '.pdf', transparent = True)
                    
                if close_figs:
                    
                    plt.close()
                    
                else:
                    
                    plt.show()
                
            
        if data_fusion_hyperpars is not None:
            
            for i in data_fusion_hyperpars['df_data_hyperpars'].keys():
                
                plt.figure()
                plt.plot(range(rounds), data_fusion_hyperpars['df_data_hyperpars'][i])
                
                plt.xlabel('Round')
                plt.ylabel('Hyperparameter ' + i)
                        
                if results_folder is not None:
                    
                    filename = results_folder + 'DFHyperpar-' + i
                
                plt.tight_layout()
                
                if saveas:
                    
                    #plt.gcf().savefig(filename + '.pdf', transparent = True)
                    plt.gcf().savefig(filename + '.png', dpi=300)
                
                if close_figs:
                    
                    plt.close()
                    
                else:
                    
                    plt.show()
                
            
        
        
def create_grid(dim = 3, domain_boundaries = [0.0,1.0], step = 0.005):
    """
    Generate a grid for the coordinates of datapoints that represent a 
    multidimensional cube of data with the desired spacing.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the datacube. The default is 3.
    domain_boundaries : [float], optional
        A list with two elements. The first element is the lower boundary of
        the domain (inclusive). The second element is the upper boundary of the
        domain (exclusive). The domain boundaries will be the same for each
        dimension. The default is [0.0,1.0].
    step : float, optional
        Step size for the points in the grid. The default is 0.005.

    Returns
    -------
    points : Numpy array
        Coordinates of the generated points. The shape of the array is
        (number of points, dim)

    """
    
    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(domain_boundaries[0], domain_boundaries[1], step)
    b = [a for i in range(dim)]
    
    grid_temp = np.meshgrid(*b, sparse=False)
    grid_temp_list = [grid_temp[i].ravel() for i in range(dim)]
    
    points = np.transpose(grid_temp_list)
    
    return points

def create_ternary_grid(step = 0.005):

    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_grid(step = step)
    points = points[abs(np.sum(points, axis=1)-1) < (step - step/5)]
    
    return points

def find_optimum(model, Y_train = None, ternary = True, 
                 domain_boundaries = [0.0, 1.0], minimize = True):
    """
    Sample a grid of points inside the domain boundaries and find the optimum
    value and location among the sampled points.

    Parameters
    ----------
    model : GPy GP regression model or GPyOpt GPModel
        The model that is sampled.
    Y_train : Numpy array, optional
        Provide the train data of the model if you want to scale the data 
        back to original units. Provide None to use the direct output of the 
        model. The default is None.
    ternary : boolean, optional
        Set to True if you want to create a grid where the elements of every
        point sum up to one (e.g., material proportions). The default is True.
    domain_boundaries : list of floats (length 2), optional
        Minimum and maximum boundaries for the domain from which the points are
        sampled from. The default is [0.0, 1.0].
    minimize : boolean, optional
        Set to True if model optimum is its minimum. Set to False if model
        optimum is its maximum. The default is True.

    Returns
    -------
    x_opt : Numpy array
        DESCRIPTION.
    y_opt : Numpy array
        DESCRIPTION.

    """
    
    if ternary is True:
        
        # Assume ternary grid.
        points = create_ternary_grid()
        
    else:
        
        # Create a full grid according to the dimensionality of the model.
        # NOTE: Does not follow constraints,
        points = create_grid(dim = model.X.shape[1], 
                             domain_boundaries = domain_boundaries)
        
    # Assumes single-task y.
    y, _ = predict_points(model, points, Y_data=Y_train)
    
    if minimize is True:
        
        idx_opt = np.argmin(y)
        
    else:
        
        idx_opt = np.argmax(y)
        
    y_opt = y[idx_opt, :]
    x_opt = points[[idx_opt], :]

    return x_opt, y_opt
        
def create_optima_arrays(BO_objects, X_accum, Y_accum, rounds, materials,
                              ternary, domain_boundaries):
    
    # Minimum value vs rounds (from the samples).
    optimum = np.full((rounds, len(materials) + 1), np.nan)
    # Minimum value vs rounds (from the model).
    model_optimum = np.full((rounds, len(materials) + 1), np.nan)    
    
    for i in range(rounds):

        idx = np.argmin(Y_accum[i], axis=0)
        opt = Y_accum[i][idx, 0]
        loc = X_accum[i][idx, :]

        optimum[i, 0:len(materials)] = loc
        optimum[i, -1] = opt

    
    # Model optimum for unconstrained and constrained space.
    for i in range(rounds):
        
        model_optimum[i,0:-1], model_optimum[i,-1] = find_optimum(BO_objects[i].model.model, # GPRegression model 
                                                    Y_train = Y_accum[i],
                                                    ternary = ternary, 
                                                    domain_boundaries = 
                                                    domain_boundaries)
        
    return optimum, model_optimum

def bo_sim_target(#bo_ground_truth_model_path='./Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222',
                  targetprop_data_source,
                  human_data_source = None,
                  materials=['CsPbI', 'MAPbI', 'FAPbI'], rounds=10,
                  init_points=None, batch_size=1,
                  acquisition_function='EI', acq_fun_params=None,
                  df_data_coll_params=None, no_plots=False,
                  results_folder='./Results/', noise_target = 1,
                  seed = None, save_memory = True, close_figs = True):
    '''
    Simulates a Bayesian Optimization cycle using the Gaussian Process
    Regression model given in bo_ground_truth_model_path as the ground truth for
    sampling BO target property. NOTE: These functions assume that the ground
    truth model contains non-scaled Y data as an argument. GPyOpt BO does not
    produce this automatically for its surrogate model - it needs to be added
    manually.

    Acquisition function options are the ones available by default in GPyOpt,
    additionally, there is 'EI_DF'. With EI_DF, one MUST deliver the correct
    acq_fun_params, otherwise the code fails. This option will use the
    acq_fun_params for building a Gaussian Process Regression model
    that will be used as the basis for data fusion -aided acquisition.

    Explanation for acq_fun_params: Three-element vector assumed with
    parameters built using ei_df_param_builder() as the first element,
    the name of the data fusion property as the second element (valid options
    are 'dft', 'visual', or 'cutoff'; 'cutoff' implementation has not been
    tested),
    a string defining how the data fusion data will be used as the third
    element (valid options are 'files' i.e. forward the data fusion data to
    EI_DF directly, which is to be used with IRL data; and 'model_none'/
    'model_all'/'model_necessary' which all fit a GPR model on all the data
    provided to serve as a ground truth source, and sample from it never/all
    the samples/when the algo defines it necessary).
    '''

    ###########################################################################
    # HARD-CODED SETTINGS (CONSIDER TURNING INTO FUNCTION PARAMS)
    
    # The sum of the amount of each material will be
    composition_total = [0.995, 1]
    # limited between these values. If you need everything to sum up to 100% within
    # 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
    # long with this value, choose [0.99,1] or even wider range.

    # Input parameter values are within this range (all the dimensions).
    domain_boundaries = [0, 1]

    # These variables are related to the Bayesian optimization.
    num_cores = 1  # Not a parallel run
    
    if type(targetprop_data_source) is GPy.models.gp_regression.GPRegression:
        
        ## No real experiments but data is queried online from a function.
        simulated_bo = True
        
    else:
        
        simulated_bo = False

    ###########################################################################
    # SET BO DATA SOURCE

    if simulated_bo == True:

        gt_model_targetprop = targetprop_data_source
        
    else:
        
        data_targetprop = targetprop_data_source # Not really tested. The idea is that there is a list of csv files for each round.
    
    ###########################################################################
    # SET DATA FUSION DATA SOURCE AND PARAMETERS
    
    if df_data_coll_params is not None:

        # Do data fusion.

        if df_data_coll_params['use_model'] is True:

            if type(human_data_source) is GPy.models.gp_regression.GPRegression:
                
                ## No real experiments but data is queried online from a function.
                gt_model_datafusionprop = human_data_source
                
                # Then data_fusion_data is cleared now that it has been used for
                # fitting the model. New data will be sampled only when the BO algo
                # requires it.
                acq_fun_params['df_data'] = [pd.DataFrame(columns=acq_fun_params['df_input_var'] +
                                                          [acq_fun_params['df_target_var']
                                                           ])]
                
        else:
                
            data_datafusionprop = human_data_source # Not really tested.
                # The data fusion data is already in here #acq_fun_params['df_data'] 
                #in
                # the required format (list of DataFrames for each round).
                

            # Fit a Gaussian process regression model to simulate the
            # data fusion variable. During the BO loop, data fusion observations
            # will be observed by sampling this model with noise.

            # ALL the data fusion data will be used for fitting the model.
            #gt_model_datafusionprop = GP_model(pd.concat(acq_fun_params['df_data']),
            #                                acq_fun_params['df_target_var'],
            #                                acq_fun_params['gp_lengthscale'],
            #                                acq_fun_params['gp_variance'],
            #                                acq_fun_params['df_input_var'])

            
            
        #if acq_fun_params['df_data'] is None:
        #    
        #    # Initialize to empty.
        #    acq_fun_params['df_data'] = [pd.DataFrame(columns=acq_fun_params['df_input_var'] +
        #                                              [acq_fun_params['df_target_var']
        #                                               ])]

        

    ###########################################################################
    # INITIALIZE VARIABLES.
    
    if simulated_bo is True:
        
        # "exact_feval" is a GPyOpt BO setting related to the expected noise
        # level of the black-box function. The BO might throw errors if a 
        # noise-free target is treated, so the variable value needs to be
        # correctly set.
        if noise_target == 0:
            exact_feval = True
        else:
            exact_feval = False
    
    # Material composition needs to sum up to 1 within the accuracy defined in
    # 'composition_total'.
    if composition_total is None:
        
        constraints = None
        ternary = False
        
    elif (np.isclose(composition_total[1], 1, 
                  atol = np.abs(composition_total[1]-composition_total[0])) == True):
        
        c0, c1 = build_constraint_str(materials, composition_total)
        
        # Search constraints.
        constraints = []
        for k in range(rounds):
    
            constraints.append([{'name': 'constr_1', 'constraint': c0},
                                {'name': 'constr_2', 'constraint': c1}])
            
        ternary = True
        
    else:
        
        ternary = True
        raise Exception('The constraint requested has not been implemented or tested.')

    # Boundaries of the search domain.
    bounds = []
    for j in range(len(materials)):
        bounds.append({'name': materials[j], 'type': 'continuous',
                       'domain': domain_boundaries})

    # A dummy parameter at the moment; will be needed in IRL BO to determine
    # if target variable values are being read from a function or from a table
    # provided by the user.
    function = True

    ###
    # Variables related to the optimization target variable:

    # Requests for the next locations to be sampled.
    x_next = [None for j in range(rounds)]
    # Same in nice dataframes.
    x_next_df = [None for j in range(rounds)]

    # Data actually collected during the round in question (in DataFrames);
    # could in experiments be different than the ones actually requested at the
    # end of the previous round.
    X_rounds = [None for j in range(rounds)]
    Y_rounds = [None for j in range(rounds)]
    # All the data actully collected by that round (in numpy arrays for GPyOpt):
    X_accum = [None for j in range(rounds)]
    Y_accum = [None for j in range(rounds)]
    # BO objects for each BO round (fitted using data acquired by the round in
    # question).
    BO_objects = [None for j in range(rounds)]
    
    
    if simulated_bo == False:

        # The actually collected optimization target points from the previous
        # rounds exist already and can be filled in to the variables.
        raise Exception('Not implemented!')

    lengthscales = [np.nan for j in range(rounds)]
    variances = [np.nan for j in range(rounds)]
    max_gradients = [np.nan for j in range(rounds)]
    gaussian_noises = [np.nan for j in range(rounds)]

    ###
    # Variables related to data fusion.

    if df_data_coll_params is not None:

        # Requests for the next locations to be queried (in DataFrames).
        data_fusion_x_next = [None for j in range(rounds)]

        # Data actually collected during the round in question (in DataFrames);
        # could in experiments be different than the ones actually requested at the
        # end of the previous round. These are
        data_fusion_XZ_rounds = [None for j in range(rounds)]
        # All the data actully collected by that round (in DataFrames):
        data_fusion_XZ_accum = [None for j in range(rounds)]
        # GP models for each BO round (fitted using data acquired by the round
        # in question, note that object is added only when the model
        # changes after having acquired more data fusion data - this is to save
        # memory):
        data_fusion_models = [None for j in range(rounds)]
        
        data_fusion_lengthscales = [None for j in range(rounds)]
        data_fusion_variances = [None for j in range(rounds)]
        data_fusion_gaussian_noises = [None for j in range(rounds)]
                
        if df_data_coll_params['use_model'] == False:

            # The actually collected data fusion points from the previous
            # rounds do already exist and can be filled in to the variables.

            for j in range(len(acq_fun_params['df_data'])):

                data_fusion_XZ_rounds[j] = acq_fun_params['df_data'][j].copy()

                # Note that acq_fun_param[df_data] will be wiped multiple times
                # during the BO loop. Data is not lost because it has been
                # saved into data_fusion_XZ_rounds here.

                # Now, the first round has at minimum been initialized to empty
                # DataFrames. There may still be 'None' slots in
                # data_fusion_XZ_rounds.
                
            for k in range(rounds):
                
                if data_fusion_XZ_rounds[k] is None:
                    
                    # Data would have already been added in the previous loop
                    # if there would have been any. Fill in an empty DataFrame.
                    data_fusion_XZ_rounds[k] = pd.DataFrame(
                        columns=acq_fun_params['df_input_var'] +
                        [acq_fun_params['df_target_var']])
                    
                # TO DO: This part works but does unnecessary redefinitions.
                # Clean up and test.
                data_fusion_XZ_accum = fill_accum_df_with_this_round(
                            data_fusion_XZ_accum, data_fusion_XZ_rounds, k)
                

    ###############################################################################
    # BEGIN BAYESIAN OPTIMIZATION

    for k in range(rounds):

        if (function == True):
            
            # Query target variable values from the provided ground truth model
            # and update X_rounds, Y_rounds, X_accum, Y_accum in place.
            X_rounds, Y_rounds, X_accum, Y_accum = query_target_data_from_model(
                k, X_rounds, Y_rounds, X_accum, Y_accum, init_points, x_next,
                gt_model_targetprop, rounds, materials, noise_level = noise_target,
                seed = seed)
            
            if df_data_coll_params is not None:
                
                # Do data fusion. Start by querying data fusion data from the model
                
                data_fusion_XZ_rounds, data_fusion_XZ_accum = query_data_fusion_data_from_model(
                    k, data_fusion_XZ_rounds, data_fusion_XZ_accum, init_points,
                    data_fusion_x_next, gt_model_datafusionprop, rounds, materials, 
                    acq_fun_params, noise_level = df_data_coll_params['noise_df'],
                    seed = seed)
            
                # Save the data fusion data for this round to the params. #that will be sent to the BO.
                acq_fun_params['df_data'] = data_fusion_XZ_accum[k]
                
        if df_data_coll_params is not None:
            
            if k == 0:
                
                # Create a data fusion model with the data fusion data collected this far.
                current_df_model = GP_model(data_fusion_XZ_accum[k],
                                            data_fusion_target_variable = acq_fun_params['df_target_var'],
                                            lengthscale = None,#acq_fun_params['gp_lengthscale'],
                                            variance = None,#acq_fun_params['gp_variance'],
                                            noise_variance = None,
                                            data_fusion_input_variables = acq_fun_params['df_input_var'],
                                            domain_boundaries = domain_boundaries)
                
                
            else:
                
                if data_fusion_XZ_rounds[k].empty is False:
                    
                    # Update existing data fusion model with the new observations.
                    
                    current_df_model.set_XY(X = acq_fun_params['df_data'][
                        acq_fun_params['df_input_var']].values,
                                            Y = acq_fun_params['df_data'][[
                                                acq_fun_params['df_target_var']]].values)
                    
                    init_hyperpars_df, lims_kernel_var_df, lims_noise_var_df = evaluate_GP_model_constraints(
                        Y = current_df_model.Y, 
                        noise_variance = None, 
                        kernel_variance = None,#acq_fun_params['gp_variance'], 
                        lengthscale = None,#acq_fun_params['gp_lengthscale'],
                        domain_boundaries = domain_boundaries)
                    
                    constrain_optimize_GP_model(current_df_model, 
                                                init_hyperpars = init_hyperpars_df, 
                                                lims_kernel_var = lims_kernel_var_df,
                                                lims_noise_var = lims_kernel_var_df)
                    
            #if (save_memory is False) or (k<2) or (no_plots == False):
                
            data_fusion_models[k] = current_df_model.copy()
            
            # Add the model in any case to the acquisition parameters.
            acq_fun_params['df_model'] = current_df_model.copy()
        
        # Define and fit BO object.
        # f=None because this code will be adapted in future for experimental
        # BO cycles.
        BO_objects[k] = GPyOpt.methods.BayesianOptimization(f=None,
                                                            domain=bounds,
                                                            constraints=constraints[k],
                                                            acquisition_type=acquisition_function,
                                                            normalize_Y=True,
                                                            X=X_accum[k],
                                                            Y=Y_accum[k],
                                                            evaluator_type='sequential',#'local_penalization',
                                                            batch_size=batch_size,
                                                            acquisition_jitter=acq_fun_params['jitter'],
                                                            acq_fun_params=acq_fun_params,
                                                            #noise_var = #10e-12,#0.1*(Y_accum[k]/Y_accum[k].max()).var(),#1e-12,#BOSS default (self.noise) is 10e-12, note that Emma needs to change this to noisy case. 0.1*(Y_accum[k]/Y_accum[k].max()).var(), #10e-12,# # GPyOpt assumes normalized Y data at the point when variance is defined.
                                                            #optimize_restarts = 10,#10,#2,
                                                            #max_iters = 2000,#1000,
                                                            exact_feval = exact_feval,
                                                            ARD=False,
                                                            kernel = None #GPy.kern.RBF#input_dim=3, ARD = True)#, 
                                                                                    # variance = 54468035, 
                                                                                    # lengthscale = 0.08)
                                                            #num_cores = 1
                                                            )

        # Suggest next points (samples to prepare).
        x_next[k] = BO_objects[k].suggest_next_locations()
        # Same as a list of DataFrames for the convenience of the user.
        x_next_df[k] = pd.DataFrame(x_next[k], columns=materials)

        # Fetch surrogate model parameters to be forwarded for the user and, if
        # data fusion is enabled, used in evaluating if data fusion data should
        # be queried.
        lengthscales[k] = BO_objects[k].model.model.kern.lengthscale.values
        variances[k] = BO_objects[k].model.model.kern.variance[0]
        gradients = BO_objects[k].model.model.predictive_gradients(x_next[k])[
            0][:, :, 0]
        gaussian_noises[k] = BO_objects[k].model.model.Gaussian_noise.variance[0]

        # Maximum gradient element value to any direction of the search space
        # for each x_next point.
        max_gradients[k] = np.max(np.abs(gradients), axis=1)
        
        current_surrogate_model_params = {'lengthscale': lengthscales[k],
                                          'variance': variances[k],
                                          'max_gradient': max_gradients[k],
                                          }
        
        if df_data_coll_params is not None:

            # Do data fusion.

            # Estimate if data fusion should be requested for (some of) the
            # suggestions for the next round.
            data_fusion_x_next[k] = determine_data_fusion_points(
                data_fusion_XZ_accum, df_data_coll_params, acq_fun_params,
                x_next, current_surrogate_model_params, materials, bounds, k)
            
            data_fusion_lengthscales[k] = current_df_model.kern.lengthscale.values
            data_fusion_variances[k] = current_df_model.kern.variance[0]
            data_fusion_gaussian_noises[k] = current_df_model.Gaussian_noise.variance[0]
            
        
    
    ###########################################################################
    # DATA TREATMENT, PLOTTING, SAVING
    
    print('Before plots and saves:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    message = 'Last suggestions for the next sampling points: ' + str(x_next[-1])
    #logging.log(21, message)
    
    # Save the model as an backup
    # dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab')
    # pickle.dump([BO_batch, x_next_df, x_next, X_rounds, Y_rounds], dbfile)
    # dbfile.close()

    surrogate_model_params = {'lengthscales': lengthscales,
                              'variances': variances,
                              'max_gradients': max_gradients,
                              'gaussian_noise_variances': gaussian_noises}

    
    if df_data_coll_params is not None:

        data_fusion_params = {'df_data_rounds': data_fusion_XZ_rounds,
                              'df_data_accum': data_fusion_XZ_accum,
                              'df_data_hyperpars': {'df_data_lengthscales': data_fusion_lengthscales,
                              'df_data_variances': data_fusion_variances,
                              'df_data_gaussian_noise_variances': data_fusion_gaussian_noises}
                              }
        
    else:

        data_fusion_params = None
    
    optimum, model_optimum = create_optima_arrays(BO_objects, X_accum, Y_accum, 
                                                  rounds, materials, ternary, 
                                                  domain_boundaries)

    if no_plots == False:

        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
        
        
        # Plot ternary-specific plots.
        plotBO(rounds, x_next_df, BO_objects, materials, X_rounds, Y_rounds,
               Y_accum, X_accum, x_next, limit_file_number=True,
               time_str=time_now, results_folder=results_folder,
               minimize = True, close_figs = close_figs)

        if acquisition_function == 'EI_DF':

            # Plot ternary-specific plots regarding data fusion.
            plotDF(rounds, materials, data_fusion_models,
                   data_fusion_XZ_accum, acq_fun_params['df_target_var'],
                   #acq_fun_params['gp_lengthscale'],
                   #acq_fun_params['gp_variance'],
                   #acq_fun_params['gp_noise_variance'],
                   acq_fun_params['p_beta'], acq_fun_params['p_midpoint'],
                   limit_file_number=True, time_str=time_now,
                   results_folder=results_folder, close_figs = close_figs)
        
            message = ('Data fusion:\nGaussian noise variances in this run: ' + 
                       str(np.mean(data_fusion_gaussian_noises)) + '+-' + 
                       str(np.std(data_fusion_gaussian_noises)) + '\n' +
                       'Lenghthscales in this run: ' + 
                       str(np.mean(data_fusion_lengthscales, axis=0)) +  '+-' + 
                       str(np.std(data_fusion_lengthscales)) + '\n' +
                       'Variances in this run: ' + 
                       str(np.mean(data_fusion_variances))  + '+-' + 
                       str(np.std(data_fusion_variances)) + '\n' +
                       'Results are saved into the given folder.')
            #logging.log(21, message)
            print(message)
            
            
        # Plots that work with any di<mensionality.
        plot_basic(Y_accum, X_accum, optimum, 
                   model_optimum, rounds, 
                   time_str = time_now, 
                   results_folder = results_folder, saveas = not no_plots,
                   hyperpars = surrogate_model_params,
                   data_fusion_hyperpars = data_fusion_params, 
                   close_figs = close_figs)
        
    message = ('Target property:\nGaussian noise variances in this run: ' + 
               str(np.mean(gaussian_noises)) + '+-' + 
               str(np.std(gaussian_noises)) + '\n' +
               'Lenghthscales in this run: ' + str(np.mean(lengthscales, axis=0)) +  '+-' + 
               str(np.std(lengthscales)) + '\n' +
               'Variances in this run: ' + str(np.mean(variances))  + '+-' + 
               str(np.std(variances)) + '\n' +
               'Max gradients in this run: ' + str(np.mean(max_gradients)) + '+-' + 
               str(np.std(max_gradients)) +  '\n' +
               'Results are saved into the given folder.')
    #logging.log(21, message)
    print(message)
                                                        
    # Not sure if these are needed anymore. I used to have some memory issues
    # that seemed to have gotten fixed by adding these, did not debug at the
    # time.
    next_suggestions = x_next_df.copy()
    optimum = optimum.copy()
    X_rounds = X_rounds.copy()
    Y_rounds = Y_rounds.copy()
    #BO_objects = BO_objects.copy()
    
    #logging.log(21, 'Jitter: ' + str(acq_fun_params['jitter']))
    #logging.log(21, 'Y values: ' + str(Y_rounds))
    
    print('After plots and saves:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    plt.close()
    
    print('After closing figs:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    
    if (save_memory is True):
        
        #BO_objects[0:(-1)] = [None] * (len(BO_objects) - 1)
        BO_objects = [None] * (len(BO_objects))
        
        if acquisition_function == 'EI_DF':
            
            #data_fusion_models[0:(-1)] = [None] * (len(data_fusion_models) - 1)
            data_fusion_models = [None] * (len(data_fusion_models))
        
    print('After clear-outs:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    
    return next_suggestions, optimum, model_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, BO_objects
