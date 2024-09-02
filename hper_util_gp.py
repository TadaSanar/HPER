#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:58:35 2024

@author: atiihone
"""

import GPy
import GPyOpt
import pickle
import numpy as np
from random import normalvariate

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

    #if seed is not None:
    #    np.random.seed(seed)
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
    posterior_mean_noisy = normalvariate(posterior_mean, 
                                 np.sqrt(gaussian_noise_variance)*noise_level)
        
        #np.random.normal(
        #posterior_mean, np.sqrt(gaussian_noise_variance)*noise_level)#np.sqrt(posterior_var)*noise_level)
    
    print('\nPredict points noisy: ', posterior_mean_noisy, posterior_mean, 
          np.sqrt(gaussian_noise_variance[0]), noise_level, '\n')
    #logging.log(21, 'Noise level: ' + str(noise_level))
    #logging.log(21, 'Posterior mean: ' + str(posterior_mean))
    #logging.log(21, 'Posterior mean noisy: ' + str(posterior_mean_noisy))
    #logging.log(21, 'Seed: ' + str(np.random.get_state()[1][0]))
    
    return posterior_mean_noisy, posterior_var, posterior_mean

def GP_model(data_fusion_data, data_fusion_target_variable = 'dGmix (ev/f.u.)', 
             lengthscale = 0.3, variance = 1, noise_variance = None,
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
        
        # The upper bound is set to the noise level that is clearly higher than
        # the maximum Y value in the dataset.
        noise_var_upper_limit = noise_var + 100*np.max(np.abs(Y))
        
        if noise_var_lower_limit > noise_var_upper_limit:
            
            noise_var_upper_limit += noise_var_lower_limit
        
        # Also kernel variance is limited because the model sometimes converged 
        # into ridiculous kernel variance values with small number of 
        # datapoints and no bounds on variance.
        kernel_var_lower_limit = kernel_var * 0.01
        kernel_var_upper_limit = kernel_var + 100*np.max(np.abs(Y))
        
        if kernel_var_lower_limit > kernel_var_upper_limit:
            
            kernel_var_upper_limit += kernel_var_lower_limit
        
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

def extract_gpmodel_params(gpmodel):
    
    lengthscale = gpmodel.kern.lengthscale.values
    variance = gpmodel.kern.variance[0]
    gaussian_noise = gpmodel.Gaussian_noise.variance[0]
    
    return lengthscale, variance, gaussian_noise


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


'''
def extract_data_fusion_model_params(current_df_model):
    
    data_fusion_lengthscale = current_df_model.kern.lengthscale.values
    data_fusion_variance = current_df_model.kern.variance[0]
    data_fusion_gaussian_noise = current_df_model.Gaussian_noise.variance[0]
    
    return data_fusion_lengthscale, data_fusion_variance, data_fusion_gaussian_noise

'''