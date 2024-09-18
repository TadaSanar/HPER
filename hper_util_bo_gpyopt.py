#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:05:09 2024

@author: atiihone
"""

import GPyOpt
from GPy.kern import RBF
import numpy as np

from hper_util_gp import extract_gpmodel_params

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

def run_bo(X, Y, bounds, constraints, acquisition_function, acq_fun_params,
           batch_size, exact_feval):
    
    if 'jitter' in acq_fun_params.keys():
        
        jitter = acq_fun_params['jitter']
        
    else:
        
        jitter = None
        
    BO_object = GPyOpt.methods.BayesianOptimization(f=None, # f=None because this code will be adapted in future for experimental BO cycles.
                                                    domain=bounds,
                                                    constraints=constraints,
                                                    acquisition_type=acquisition_function,
                                                    normalize_Y=True, # The predict() function implemented in this file assumes that BO has been implemented with zero mean unit std normalization (default in GPyOpt). Modify the predict function if this is not the case.
                                                    X=X,
                                                    Y=Y,
                                                    evaluator_type='local_penalization',
                                                    batch_size=batch_size,
                                                    acquisition_jitter=jitter,
                                                    acq_fun_params=acq_fun_params,
                                                    #noise_var = #10e-12,#0.1*(Y_accum[k]/Y_accum[k].max()).var(),#1e-12,#BOSS default (self.noise) is 10e-12, note that Emma needs to change this to noisy case. 0.1*(Y_accum[k]/Y_accum[k].max()).var(), #10e-12,# # GPyOpt assumes normalized Y data at the point when variance is defined.
                                                    #optimize_restarts = 10,#10,#2,
                                                    #max_iters = 2000,#1000,
                                                    exact_feval = exact_feval,
                                                    ARD=False,
                                                    kernel = RBF#input_dim=3, ARD = True)#, 
                                                    # variance = 54468035, 
                                                    # lengthscale = 0.08)
                                                    #num_cores = 1
                                                    #acquisition_optimizer = 'DIRECT'
                                                    )
    
    # Suggest next points (samples to prepare).
    x_next = suggest_next_samples(BO_object)
    
    # Fetch surrogate model parameters to be forwarded for the user and, if
    # data fusion is enabled, used in evaluating if data fusion data should
    # be queried.
    lengthscale, variance, gaussian_noise = extract_gpmodel_params(
        BO_object.model.model)
    
    # Maximum gradient element value to any direction of the search space
    # for each x_next point.
    gradients = BO_object.model.model.predictive_gradients(x_next)[0][:, :, 0]
    max_gradient = np.max(np.abs(gradients), axis=1)
    
    current_surrogate_model_params = {'lengthscale': lengthscale,
                                      'variance': variance,
                                      'max_gradient': max_gradient,
                                      'gradients': gradients,
                                      'gaussian_noise': gaussian_noise
                                          }
    
    return BO_object, x_next, current_surrogate_model_params

def suggest_next_samples(BO_object):
    
    x_next = BO_object.suggest_next_locations()
    
    return x_next
