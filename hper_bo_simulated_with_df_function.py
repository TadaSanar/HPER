"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

#Libraries: seaborn, scipy, pandas, Python 3.XX and GPyOpt are required
#import sys
#sys.path.insert('./GPyOpt_DFT/GPyOpt_DFT/')

import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns
import GPyOpt
from GPyOpt.acquisitions.EI_DFT import GP_model
from scipy.integrate import simps

import ternary
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
from RGB_data import RGB_data

from plotting_v2 import triangleplot
from plotting_v2 import plotBO

def tolerance_factor(suggestion_df = None, tolerance_factor_bound = None):
    if (suggestion_df is not None and tolerance_factor_bound is None):
        # Calculate the tolerance factor for the given dataframe of suggestions.
        tolerance_factor_series = (220+(167*suggestion_df.CsPbI +
                                        217*suggestion_df.MAPbI +
                                        253*suggestion_df.FAPbI))/(1.4142*(119+220))
        # Works for materials = ['CsPbI', 'MAPbI', 'FAPbI'] only!
        result = tolerance_factor_series
    elif (suggestion_df is None and tolerance_factor_bound is not None):
        tolerance_factor_constraint = str(tolerance_factor_bound) + ' - ((220+(167*x[:,0] + 217*x[:,1] + 253*x[:,2]))/(1.4142*(119+220)))' # This function is designed for these elements in this
        # specific order only: materials = ['CsPbI', 'MAPbI', 'FAPbI']
        result = tolerance_factor_constraint
    else:
        raise ValueError('This function is not intended to be used for this kind of inputs.')

    return result
    
def f_booth(x):
    
    x1 = x[:,0]
    x2 = x[:,1]
    
    f = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

    return f

def predict_points(GP_model, x_points, Y_train = None):

    posterior_mean, posterior_std = GP_model.predict(x_points)
    
    # Normally, a GP model is trained with data that has not been scaled previously.
    # If the model has been trained on data that has already been scaled to
    # zero mean, unit variance (e.g., GPyOpt BayesianOptimization), original
    # train data points, Y_train, are needed for scaling the predictions back
    # to the original units.
    if Y_train is not None: # The model has been trained on data that is not scaled.
        # These values are required for scaling the model prediction back to the original units.
        train_data_mean = np.mean(Y_train)
        train_data_std = np.std(Y_train)
        
        # Scaling the normalized data back to the original units.
        posterior_mean = posterior_mean*train_data_std+train_data_mean # Predicted y values
        posterior_std = posterior_std*train_data_std # Std for the predictions        
    
    return posterior_mean, posterior_std    

def predict_points_noisy(GP_model, x_points, Y_train = None):

    # Predictions.
    posterior_mean, posterior_std = predict_points(GP_model, x_points, Y_train = Y_train)
    
    # Adding Gaussian noise to the mean predictions.
    posterior_mean_noisy = np.random.normal(posterior_mean, posterior_std)
    
    return posterior_mean_noisy, posterior_mean, posterior_std

def predict_points_noisy_c2a(GP_model, x_points, y_ground_truth):
    
    mean_noisy, mean, std = predict_points_noisy(GP_model, x_points, Y_train = y_ground_truth)
    
    return mean_noisy

def predict_points_c2a(GP_model, x_points, y_ground_truth):
    
    mean, std = predict_points(GP_model, x_points, Y_train = y_ground_truth)
    
    return mean

#%%
###############################################################################
def ei_dft_param_builder(acquisition_type,
    data_fusion_target_variable = 'dft', files = None, lengthscale = None,
    variance = None, beta = None, midpoint = None,
    data_fusion_input_variables = None):
    
    '''
    This function builds a properly formatted param dictionary for EI_DFT
    acquisition function when using bo_sim_target().
    
    The only acquisition_type implemented is 'EI_DFT'. The allowed options for
    data_fusion_target_variable are 'dft', 'visual', or 'cutoff'. If the rest
    of the variables are None, hard-coded default values listed will be resumed.
    '''
    
    
    if acquisition_type == 'EI_DFT':
        
        if data_fusion_target_variable == 'dft':
            
            if files == None:
                
                # These files contain DFT data that is integrated into the optimization loop as
                # a soft constraint (starting from the round it is first listed - if the vector
                # is shorter than the number of rounds, it is assumed that no data is being
                # added to data fusion in the rounds not listed).
                files = [['./phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
                 './phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
                 './phasestability/CsMA/fulldata/CsMA_T300_above.csv']
                 ]
                
            if data_fusion_input_variables == None:
                variable = 'dGmix (ev/f.u.)'
            if lengthscale == None: # For Gaussian process regression
                lengthscale = 0.03
            if variance == None: # For GPR
                variance = 2
            if beta == None: # For probability model
                beta = 0.025
            if midpoint == None: # For P model
                midpoint = 0 # For P

        elif data_fusion_target_variable == 'visual':
            
            if files == None:
                
                # Visual quality of the samples as a constraint.              
                files = [['./visualquality/visualquality_round_0-1.csv']]
                
            if data_fusion_input_variables == None:
                variable = 'Quality'
            if lengthscale == None: # For Gaussian process regression
                lengthscale = 0.1
            if variance == None: # For GPR
                variance = 0.1
            if beta == None: # For probability model
                beta = 0.1#0.2#25
            if midpoint == None: # For P model
                midpoint = 0 # For P    
    
        elif data_fusion_target_variable == 'cutoff':
            
            if files == None:
                
                # Arbitrary function (such as a direct cutoff) as a constraint.
                cutoff_files = [['./arbitrary_function/cutoff.csv']]
                
            if data_fusion_input_variables == None:
                variable = 'Cutoff'
            
            if lengthscale == None: # For Gaussian process regression
                lengthscale = 0.05
            if variance == None: # For GPR
                variance = 0.05
            if beta == None: # For probability model
                beta = 0.025
            if midpoint == None: # For P model
                midpoint = 0 # For P
                
        else:
            raise Exception('Data fusion target variable ' + data_fusion_target_variable + ' has not been implemented in the parameter builder. Please provide another variable name.')
        
        if (files != None):
        
            # Retrieve the data.
            data_fusion_data = []
            
            for i in range(len(files)):
        
                data_fusion_data.append([])
                
                for j in range(len(files[i])):    
                    data_fusion_data[i].append(pd.read_csv(files[i][j]))
                
                data_fusion_data[i] = pd.concat(data_fusion_data[i])
    
        acq_fun_params = [data_fusion_data, variable, lengthscale, variance,
                          beta, midpoint]
    else:
        raise Exception('The parameter builder is not required for this acquisition function. The builder has been implemented only for EI_DFT.')
        
    return acq_fun_params

def ei_dft_param2str(acq_fun_params):
    
    output_str = ('-dftarget-' + acq_fun_params[1] + '-lengthscale-' +
    str(acq_fun_params[0][2]) + '-variance-' + str(acq_fun_params[0][3]) + '-beta-' +
    str(acq_fun_params[0][4]) + '-midpoint-' + str(acq_fun_params[0][5]))
    
    return output_str

def bo_sim_target_and_fusion(bo_ground_truth_model_path = 'C2a_GPR_model_with_unscaled_ydata-20190730172222',
                             data_fusion_target_variable = 'dft',
                             materials = ['CsPbI', 'MAPbI', 'FAPbI'],
                             rounds = 10, init_points = 2, batch_size = 1,
                             no_plots = False):
    
    '''
    Simulates a Bayesian Optimization cycle using the Gaussian Process
    Regression model given in bo_ground_truth_model_paths as the ground truth for
    sampling BO target property.
    
    Additional information (e.g., human opinion or DFT information) is fed in
    via data fusion technique.
    
    Acquisition function is always EI_DFT. EI_DFT parameter chooser is
    currently implemented only for properties listed in ei_dft_param_builder().
    Other variables will result in an error when running BO.
    
    TO DO THIS FUNCTION NOT NEEDED, REMOVE
    '''
    
    acq_fun_params = ei_dft_param_builder(acquisition_type = 'EI_DFT',
        data_fusion_target_variable = data_fusion_target_variable)
    
    optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step = bo_sim_target(
        bo_ground_truth_model_path, materials = ['CsPbI', 'MAPbI', 'FAPbI'], 
        rounds = 10, init_points = 2, batch_size = 1, 
        acquisition_function = 'EI_DFT', acq_fun_params = acq_fun_params,
        no_plots = False)

    return optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step

def bo_sim_target(bo_ground_truth_model_path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222', 
                  materials = ['CsPbI', 'MAPbI', 'FAPbI'], rounds = 10,
                  init_points = 2, batch_size = 1, acquisition_function = 'EI',
                  acq_fun_params = [None, None, None], no_plots = False):

    '''
    Simulates a Bayesian Optimization cycle using the Gaussian Process
    Regression model given in bo_ground_truth_model_path as the ground truth for
    sampling BO target property. NOTE: These functions assume that the ground
    truth model contains non-scaled Y data as an argument. GPyOpt BO does not
    produce this automatically for its surrogate model - it needs to be added
    manually.
    
    Acquisition function options are the ones available by default in GPyOpt,
    additionally, there is 'EI_DFT'. With EI_DFT, one MUST deliver the correct
    acq_fun_params, otherwise the code fails. This option will use the
    acq_fun_params for building a Gaussian Process Regression model
    that will be used as the basis for data fusion -aided acquisition.
    
    Explanation for acq_fun_params: Three-element vector assumed with
    parameters built using ei_dft_param_builder() as the first element,
    the name of the data fusion property as the second element (valid options
    are 'dft', 'visual', or 'cutoff'; 'cutoff' implementation has not been
    tested),
    a string defining how the data fusion data will be used as the third
    element (valid options are 'files' i.e. forward the data fusion data to
    EI_DFT directly, which is to be used with IRL data; and 'model_none'/
    'model_all'/'model_necessary' which all fit a GPR model on all the data
    provided to serve as a ground truth source, and sample from it never/all
    the samples/when the algo defines it necessary).
    '''
    
    #original_folder = os.getcwd()
    
    # Load already existing stability data as the "ground truth" of stability.
    with open(bo_ground_truth_model_path,'rb') as f:
        stability_model = pickle.load(f)
    
    # Determine acquisition function for BO (data fusion via EI_DFT will
    # require extra parameters, the other acquisitions do not).
    if acquisition_function != 'EI_DFT':
        
        # Other acquisition functions than EI_DFT do not require the additional params.
        data_fusion_data = None
        data_fusion_variable = None
        lengthscale = None
        variance = None
        beta = None
        midpoint = None
        current_data_fusion_data = None
        
    else:
        
        if acq_fun_params[0] == None:
            
            # Define EI_DFT parameters.
            acq_fun_params[0] = ei_dft_param_builder(acquisition_type = acquisition_function,
                data_fusion_target_variable = acq_fun_params[1])
           
        [data_fusion_data, data_fusion_variable, lengthscale, variance, beta, midpoint] = acq_fun_params[0]
        # Variable 'current_data_fusion_data' is not defined here but within the 'rounds' loop below.
                
        # Define data fusion mode.
        #if acq_fun_params[2] == 'file':
            
        #    # The data is read from the 'data_fusion_data', no need for additional actions.
            
        if acq_fun_params[2][0:5] == 'model':
            
            # Fit a Gaussian process regression model to simulate the ground
            # data fusion variable. During the BO loop, data fusion observations
            # will be observed by sampling this model with noise.
            
            # ALL the data fusion data will be used for fitting the model.
            data_fusion_gt_model = GP_model(pd.concat(data_fusion_data),
                                            data_fusion_variable,
                                            lengthscale, variance,
                                            materials)
            
            # Then data_fusion_data is cleared. New data will be sampled only
            # when the BO algo requires it.
            data_fusion_data = [pd.DataFrame(columns = materials + [data_fusion_variable])]
            
            print("Data fusion model created.")

    
   ###############################################################################
    
    # Collect the data and compute the figure of merit.
    
    constraints = [None for j in range(rounds)]
    
    # Define the variables and the domain for each
    # One can define here also material- or round-specific parameters.
    bounds = [None for j in range(len(materials))]
    
    # Data collected during that round:
    X_rounds = [None for j in range(rounds)]
    Y_rounds = [None for j in range(rounds)]
    # All the data collected by that round:
    X_step = [np.empty((0,len(materials))) for j in range(rounds)]
    Y_step = [np.empty((0,1)) for j in range(rounds)] # Change (0,1) to (0,2) if multiobjective
    
    
    x_next = [None for j in range(rounds)] # Suggestions for the next locations to be sampled.
    suggestion_df = [None for j in range(rounds)] # Same in nice dataframes.
    
    BO_batch_model = [None for j in range(rounds)] # Batch BO objects for each BO round (with data acquired by that time).
    
    
    # The following is for IRL BO, commented out but not removed since it will
    # be required later on.
    '''
    if function is False:
        df_compositions = [None for j in range(rounds)]
        mean_RGB = [None for j in range(rounds)]
        red = [None for j in range(rounds)]
        blue = [None for j in range(rounds)]
        green = [None for j in range(rounds)]
        times = [None for j in range(rounds)]
        merit_area = [None for j in range(rounds)]
        merit_diff = [None for j in range(rounds)]
        merit_inv_moment = [None for j in range(rounds)]
        degradation_input = [None for j in range(rounds)]
        compositions_input = [None for j in range(rounds)]
        
        for k in range(rounds):
            df_compositions[k] = pd.read_csv(folders[k] + 'Samples.csv')
            #Import RGB data and sample details.
            mean_RGB[k] = RGB_data(folders[k], df_compositions[k].get("Sample"), cutoff, is_calibrated, is_rgb)
            #Get dataframes with sample as level
            red[k], blue[k], green[k], times[k]= mean_RGB[k].preprocess()
            
            #Compute figure-of-merit, returns dataframe with figure of merit
            merit_area[k] = mean_RGB[k].compute_degradation(method = 'area')
            merit_diff[k] = mean_RGB[k].compute_degradation(method = 'diff_area')
            merit_inv_moment[k] = mean_RGB[k].compute_degradation(method = 'inverted_moment')
            
            #Pick the one that looks best, in this case will do merit_diff_abs
            #Drop the empty slots and prepare the data to be fed into GpyOpt
            print('Round ' + str(k) + ':')
            print('These slots are empty:')
            print(merit_diff[k][merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)])
            degradation_input[k] = merit_diff[k][~merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
            compositions_input[k] = df_compositions[k][~df_compositions[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
            
            #Creating dataframe to report comparison between methods
            merit_diff[k] = merit_diff[k][~merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
            merit_inv_moment[k] = merit_inv_moment[k][~merit_inv_moment[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
            merit_area[k] = merit_area[k][~merit_area[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
    
            composition_total = [0.995, 1] # The sum of the amount of each material will be
            # limited between these values. If you need everything to sum up to 100% within
            # 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
            # long with this value, choose [0.99,1] or even wider range. The code currently
            # works only for exactly three materials.
            tolerance_factor_bound = 0.80 # Tolerance factor will be limited above this value.
            tolerance_factor_function = tolerance_factor(suggestion_df = None, 
                                                         tolerance_factor_bound = tolerance_factor_bound)
            
            constraints[k] = [{'name': 'constr_1', 'constraint': 'x[:,0] +x[:,1] + x[:,2] - ' + str(composition_total[1])},
                           {'name': 'constr_2', 'constraint': str(composition_total[0])+'-x[:,0] -x[:,1] - x[:,2] '},
                           {'name': 'constr_3', 'constraint': tolerance_factor_function}]
            if n_batch == None:
                # The batch size i.e. the number of suggestions the algorithm gives is the
                # same as the number of samples that were degraded in the first round.
                batch_size[k] = len(degradation_input[0])
            else:
                batch_size[k] = n_batch
            
        for j in range(len(materials)):
            bounds[j] = {'name': materials[j], 'type': 'continuous', 'domain': (0,1)}
    '''
    
    composition_total = [0.995, 1] # The sum of the amount of each material will be
    # limited between these values. If you need everything to sum up to 100% within
    # 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
    # long with this value, choose [0.99,1] or even wider range.
        
    # TO DO.
    # This currently works only for exactly three materials.
    tolerance_factor_bound = 0.80 # Tolerance factor will be limited above this value.
    tolerance_factor_function = tolerance_factor(suggestion_df = None, 
                                                 tolerance_factor_bound = tolerance_factor_bound)
    
    for k in range(rounds):
        
        # TO DO Poista käytöstä.
        # TO DO: Generalize to more than three materials or shift up to the user defined part.
        constraints[k] = [{'name': 'constr_1', 'constraint': 'x[:,0] +x[:,1] + x[:,2] - ' + str(composition_total[1])},
                           {'name': 'constr_2', 'constraint': str(composition_total[0])+'-x[:,0] -x[:,1] - x[:,2] '}#,
                           #{'name': 'constr_3', 'constraint': tolerance_factor_function}
                           ]
        
    for j in range(len(materials)):
        bounds[j] = {'name': materials[j], 'type': 'continuous', 'domain': (0,1)}
    
    # To do, is this really still needed?
    #os.chdir(original_folder)
    
    # These variables are related to the Bayesian optimization.
    num_cores = 1 # Not a parallel run
    jitter = 0.01 # The level of exploration.
    

    ###############################################################################
    #%%
    # BEGIN BAYESIAN OPTIMIZATION
    
    # Will be needed in IRL BO.
    function = True
    
    lengthscales = []
    variances = []
    max_gradients = []
    
    for k in range(rounds):
            
        # The implementation is like this because it is compatible with IRL data fusion BO. 
        if acquisition_function == 'EI_DFT':

            # Collect the data fusion data observed by this round (cumulatively).
            if len(data_fusion_data) > k:
            
                if k == 0:
                    current_data_fusion_data = data_fusion_data[k].copy()
                else:
                    current_data_fusion_data = pd.concat([
                        current_data_fusion_data, data_fusion_data[k]])

        
        if (k == 0) and (function == True):
            # Initialization with the given grid points.
            grid_init = np.array(init_points)        
            compositions_input = [pd.DataFrame(grid_init, columns = materials)]
            degradation_input = []
            
        if (k > 0) and (function == True):
            # The locations suggested after the previous BO round will be
            # sampled in this round.
            compositions_input.append(pd.DataFrame(x_next[k-1],
                                                   columns = materials))
            
        df = compositions_input[k].copy()
        
        '''
        # These lines perform the selected operations to the duplicate samples
        # (choose first, choose last, choose the average, do nothing).
        if function == False:
            
            df['Merit'] = degradation_input[k]['Merit'].values
            if duplicate_operation == 'first':
                df = df.drop_duplicates(subset=materials, keep='first').reset_index()
            elif duplicate_operation == 'last':
                df = df.drop_duplicates(subset=materials, keep='last').reset_index()
            elif duplicate_operation == 'full':
                df = df
            elif duplicate_operation == 'mean':
                df = df.groupby(materials).mean().reset_index()
            else:
                raise Exception('The given value for treating duplicate samples is not valid. Give a valid value.')
        '''
        
        # Function predictions. Each round, the BO model is trained from zero.
        # This is not computationally efficient, though. Should we modify at
        # some point? Or do we want to keep the BO model for each round intact?
        # TO DO: clean up so that works for any #D
        if function == True:
            x = df.iloc[:,0:len(materials)].values
            df['Merit'] = predict_points_noisy_c2a(stability_model, x, stability_model.model.Y)#predict_points_noisy_c2a(stability_model, x, stability_model.model.Y)
            degradation_input.append(df)
            
        
        # X is a vector of compositions, Y is a vector of merit values.
        X_rounds[k] = df[materials].values
        # Reshaping is done to go from (n,) to (n,1), which is required by GPyOpt.
        Y_rounds[k] = np.reshape(df['Merit'].values, (df['Merit'].values.shape[0], 1))
        
        # For each BayesianOpt round, we include only the data that has been
        # collected by that time.
        for j in range(rounds):
            if j >= k:
                X_step[j] = np.append(X_step[j], X_rounds[k], axis=0)
                Y_step[j] = np.append(Y_step[j], Y_rounds[k], axis=0)
        
        # Do the Bayesian Optimization.
        #print('X_step and Y_step for round ' + str(k) + ':', X_step[k], Y_step[k])
        #Define Bayesian Opt object
        #Instantiate BO object, f=None as we are using tabular data (to ), no analytical function
        BO_batch_model[k] = GPyOpt.methods.BayesianOptimization(f=None,#f_booth,  
                                                domain = bounds,
                                                constraints = constraints[k],
                                                acquisition_type = acquisition_function, #  'EI_DFT' or 'EI'
                                                data_fusion_data = current_data_fusion_data,
                                                data_fusion_target_variable = data_fusion_variable,
                                                normalize_Y = True,
                                                X = X_step[k],
                                                Y = Y_step[k],
                                                evaluator_type = 'local_penalization',
                                                batch_size = batch_size,
                                                acquisition_jitter = jitter,
                                                lengthscale = lengthscale,
                                                variance = variance,
                                                beta = beta,
                                                midpoint = midpoint,
                                                data_fusion_input_variables = materials)    
        #Suggest next points (samples to prepare).
        x_next[k] = BO_batch_model[k].suggest_next_locations()
        suggestion_df[k] = pd.DataFrame(x_next[k], columns = materials)
        suggestion_df[k]['Total'] = suggestion_df[k].sum(axis = 1)
        #suggestion_df[k]['Tolerance Factor'] = tolerance_factor(
        #        suggestion_df = suggestion_df[k],
        #        tolerance_factor_bound = None)
        BO_batch_model[k].plot_acquisition() # Works only for max 2D.
        
        gradients = []
        grad_step_x = []
        for l in range(len(materials)):
            # Calculate gradient.
            # TO DO: We probably have a way to dig out the gradient directly from the model, how?
            # BO_batch[i].model.model.predictive_gradients(points)
            grad_step_x.append((bounds[l]['domain'][1] - bounds[l]['domain'][0])/100)
            
            x1 = x_next[k].copy()
            x1[:,l] = x1[:,l] - grad_step_x[l]
            x2 = x_next[k]
            y1 = BO_batch_model[k].model.predict(x1)[0]
            y2 = BO_batch_model[k].model.predict(x2)[0]
            gradients.append(((y2-y1)/(x2-x1))[:,l])
        
        grad_max = np.max(np.abs(gradients), axis = 0) # Maximum gradient element value to any direction of the search space for each point.
        lengthscale_opt = BO_batch_model[k].model.model.kern.lengthscale[0]
        variance_opt = BO_batch_model[k].model.model.kern.variance[0]
        
        max_gradients.append(grad_max)
        lengthscales.append(lengthscale_opt)
        variances.append(variance_opt)
        
        # Estimate if data fusion should be requested for the next round
        # suggestions.
        if acq_fun_params[2] == 'model_none':

            # Never sample from the model, just add an empty df records.
            data_fusion_data.append(pd.DataFrame(columns = materials + [data_fusion_variable]))
            
        elif acq_fun_params[2] == 'model_all':

            # Sample all the data points.
            data_fusion_data.append(pd.DataFrame(
                np.concatenate((x_next[k],
                                data_fusion_gt_model.predict_noiseless(
                                    x_next[k])[0]), axis=1),
                columns = materials + [data_fusion_variable])) # TO DO: Add noise!
            
        elif acq_fun_params[2] == 'model_necessary':
            
            #print(lengthscale_opt, variance_opt)
            
            # Sample only if any of the input gradients is larger than
            # kernel varience divided by its lengthscale.
            gradient_limit = np.sqrt(variance_opt)/lengthscale_opt/2
            
            #print(gradient_limit < grad_max, gradient_limit, grad_max, lengthscale_opt, variance_opt)
            
            # Sample data fusion source data either via ground truth model:
            if function is True:
                
                # Pick new points for which the surrogate model has a high
                # gradient, no matter if there is an earlier data fusion point
                # nearby.
                new_df_points_x_g = x_next[k][grad_max > gradient_limit]
                #print(new_df_points_x.shape, lengthscale_opt, variance_opt, gradient_limit, grad_max) 
                
                # Pick new points for which the surrogate model does not have
                # a high gradient but they are located far away from the
                # previously sampled points (in unknown region).
                new_df_points_x_u = x_next[k][grad_max <= gradient_limit]
                
                # Drop the points that have an earlier data fusion point nearby.
                # 'Nearby' is 5% of the domain length here.
                limit = (bounds[0]['domain'][1] - bounds[0]['domain'][0])/100*10
                index = 0
                for l in range(len(new_df_points_x_g)): # Should be u finally!
                    
                    if current_data_fusion_data.shape[0] > 0:
                    
                        #if (np.sqrt(np.sum((current_data_fusion_data.iloc[:,0:len(materials)] -new_df_points_x[index])**2, axis=1)) < 0.05).shape[0] > 0:
                        if np.any(np.sqrt(np.sum((current_data_fusion_data.iloc[:,0:len(materials)] -new_df_points_x_g[index])**2, axis=1)) < limit):
                            new_df_points_x_g = np.delete(new_df_points_x_g, index, axis=0)
                            #print('Deleted points based on exclusion.')
                
                # Combine the twp criteria.
                new_df_points_x = new_df_points_x_g #np.append(new_df_points_x_g, new_df_points_x_u, axis = 0)
                # Make predictions on the data fusion property.
                if new_df_points_x.shape[0] > 0:
                    
                    #print(new_df_points_x.shape)
                    new_df_points_y = data_fusion_gt_model.predict_noiseless(new_df_points_x)[0] # TO DO: Add noise!
                    
                    # Then add to data_fusion_data
                    data_fusion_data.append(pd.DataFrame(
                        np.concatenate((new_df_points_x, new_df_points_y),
                                       axis = 1),
                    columns = materials + [data_fusion_variable]))
                    
                else:
                    
                    new_df_points_y = np.array([])
                    data_fusion_data.append(pd.DataFrame(
                    columns = materials + [data_fusion_variable]))
                
                print('Round ' + str(k) + ': ', new_df_points_x)
                
                
            else:
                
                print('Not implemented.')

        
    
    # Plot and save the results.
    
    #if function == True:
    #    degradation_input = compositions_input
    
    
    if no_plots == False:
        plotBO(rounds, suggestion_df, compositions_input, degradation_input, BO_batch_model, materials, X_rounds, x_next, Y_step, X_step, limit_file_number = True)    
    
    print('Last suggestions for the next sampling points: ', x_next[-1])
    print('Results are saved into folder ./Results.')
    
    # Save the model as an backup
    # dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab') 
    # pickle.dump([BO_batch, suggestion_df, x_next, X_rounds, Y_rounds], dbfile)                      
    # dbfile.close()
    
    # Minimum value vs rounds.
    optimum = []
    for i in range(rounds):
        optimum.append(np.min(Y_step[i]))
    
    if no_plots == False:
            plt.figure()
            plt.plot(range(rounds), optimum)
            plt.show()


    return optimum, rounds, suggestion_df, compositions_input, degradation_input, BO_batch_model, materials, X_rounds, x_next, Y_step, X_step, data_fusion_data, lengthscales, variances, max_gradients
