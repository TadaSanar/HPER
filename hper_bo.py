"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import GPyOpt
from GPyOpt.acquisitions.EI_DFT import GP_model
import pickle
import datetime
import matplotlib.pyplot as plt
from plotting_v2 import plotBO
from plotting_data_fusion import plotDF

def predict_points(GP_model, x_points, Y_data = None):
    '''
    For a GPy GP regression model.
    '''
    posterior_mean, posterior_var = GP_model.predict(x_points)
    
    # If the model has been trained with already-scaled (zero mean, unit
    # variance) data, the provided train data 'Y_data' will be used for scaling
    # the predictions to the correct units.
    if Y_data is not None:
        posterior_mean_true_units = posterior_mean*np.std(Y_data) + np.mean(Y_data)
        posterior_var_true_units = (np.std(Y_data))**2 * posterior_var
    
        posterior_mean = posterior_mean_true_units
        posterior_var = posterior_var_true_units
        
    return posterior_mean, posterior_var

def predict_points_noisy(GP_model, x_points, Y_data = None):

    # Predictions.
    posterior_mean, posterior_var = predict_points(GP_model, x_points, Y_data = Y_data)
    
    # Adding Gaussian noise to the mean predictions.
    posterior_mean_noisy = np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    return posterior_mean_noisy, posterior_var, posterior_mean

def ei_dft_param_builder(acquisition_type, data_fusion_target_property = 'dft',
                         data_fusion_input_variables = ['CsPbI', 'MAPbI', 'FAPbI'],
                         optional_acq_params = None):
    
    '''
    This function builds a properly formatted param dictionary for EI_DFT
    acquisition function when using bo_sim_target().
    
    The only acquisition_type implemented is 'EI_DFT'. The allowed options for
    data_fusion_target_variable are 'dft', 'visual', or 'cutoff'. If the rest
    of the variables are None, hard-coded default values listed will be resumed.
    '''
    
    if acquisition_type == 'EI_DFT':
        
        # Init all params.

        list_p = ['files', 'lengthscale', 'variance',
                  'beta', 'midpoint', 'df_target_variable']
        p = {'df_input_variables': data_fusion_input_variables}
        
        for i in range(len(list_p)):
            
            p[list_p[i]] = None
                
        if optional_acq_params is not None:
            
            for i in range(len(list_p)):
                
                if list_p[i] in optional_acq_params:
                    
                    p.update({list_p[i]: optional_acq_params[list_p[i]]})
                
        
        if data_fusion_target_property == 'dft':
            
            if p['files'] is None:
                
                # These files contain DFT data that is integrated into the optimization loop as
                # a soft constraint (starting from the round it is first listed - if the vector
                # is shorter than the number of rounds, it is assumed that no data is being
                # added to data fusion in the rounds not listed).
                files = [['./phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
                 './phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
                 './phasestability/CsMA/fulldata/CsMA_T300_above.csv']
                 ]
                
            if p['df_target_variable'] == None:
                variable = 'dGmix (ev/f.u.)'
            if p['lengthscale'] == None: # For Gaussian process regression
                lengthscale = 0.03
            if p['variance'] == None: # For GPR
                variance = 2
            if p['beta'] == None: # For probability model
                beta = 0.025
            if p['midpoint'] == None: # For P model
                midpoint = 0 # For P

        elif data_fusion_target_property == 'visual':
            
            if p['files'] is None:
                
                # Visual quality of the samples as a constraint.              
                files = [['./Source_data/visualquality/visualquality_round_0-1.csv']]
                
            if p['df_target_variable'] == None:
                variable = 'Quality'
            if p['lengthscale'] == None: # For GP regression
                lengthscale = 0.1
            if p['variance'] == None: # For GPR
                variance = 0.1
            if p['beta'] == None: # For probability model
                beta = 0.1
            if p['midpoint'] == None: # For P model
                midpoint = 0 # For P    
    
        elif data_fusion_target_property == 'cutoff':
            
            if p['files'] is None:
                
                # Arbitrary function (such as a direct cutoff) as a constraint.
                files = [['./arbitrary_function/cutoff.csv']]
                
            if p['df_target_variable'] == None:
                variable = 'Cutoff'
            
            if p['lengthscale'] == None: # For Gaussian process regression
                lengthscale = 0.05
            if p['variance'] == None: # For GPR
                variance = 0.05
            if p['beta'] == None: # For probability model
                beta = 0.025
            if p['midpoint'] == None: # For P model
                midpoint = 0 # For P
                
        else:
            raise Exception('Data fusion target variable ' + p['df_target_variable'] + 
                            ' has not been implemented in the parameter builder. Please provide another variable name.')
        
        if (files != None):
        
            # Retrieve the data.
            data_fusion_data = []
            
            for i in range(len(files)):
        
                data_fusion_data.append([])
                
                for j in range(len(files[i])):    
                    data_fusion_data[i].append(pd.read_csv(files[i][j]))
                
                data_fusion_data[i] = pd.concat(data_fusion_data[i])
    
        ei_dft_params = {'df_data': data_fusion_data,
                         'df_target_var': variable,
                         'df_target_prop': data_fusion_target_property,
                         'df_input_var': p['df_input_variables'],
                         'gp_lengthscale': lengthscale,
                         'gp_variance': variance,
                         'p_beta': beta,
                         'p_midpoint': midpoint
                         }
    else:
        raise Exception('The parameter builder is not required for this acquisition function. The builder has been implemented only for EI_DFT.')
        
    return ei_dft_params

def acq_param_builder(acquisition_function, data_fusion_property = None,
                     data_fusion_input_variables = None, optional_acq_params = None):

    if data_fusion_property is None:
        
        acq_fun_params = None
        
    else:
        
        if acquisition_function == 'EI_DFT':
            
            acq_fun_params = ei_dft_param_builder(acquisition_function,
                                 data_fusion_target_property = data_fusion_property,
                                 data_fusion_input_variables = data_fusion_input_variables,
                                 optional_acq_params = optional_acq_params)
            
        else:
            
            raise Exception('Data fusion has not been implemented for this acquisition function.')
                
    return acq_fun_params

def acq_fun_param2descr(acq_fun, acq_fun_params = None):
    
    output_str = acq_fun
    
    if acq_fun == 'EI_DFT':
        
        ei_dft_params = acq_fun_params
        output_str = (output_str + '-dftarget-' + ei_dft_params['df_target_prop'] +
                      '-lengthscale-' + str(ei_dft_params['gp_lengthscale']) +
                      '-variance-' + str(ei_dft_params['gp_variance']) +
                      '-beta-' + str(ei_dft_params['p_beta']) +
                      '-midpoint-' + str(ei_dft_params['p_midpoint']))
    
    return output_str

def df_data_coll_param_builder(df_method = None, gradient_param = None, exclusion_param = None):
    
    if df_method is None:
        
        # No data fusion is done.
        df_data_coll_params = None
        
    else:
        
        if df_method.find('model') != -1:
            
            use_model = True
            
        else:
            
            use_model = False
        
        df_data_coll_params = {'use_model': use_model}
        
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
            
            raise Exception('Data fusion data collection has not been implemented for this case.')
            
        df_data_coll_params['method'] = method
    
    return df_data_coll_params
    
def df_data_coll_method_param2descr(df_data_coll_params):
    
    output_str = ''
    
    if df_data_coll_params is not None:
        
        if df_data_coll_params['use_model']:
            
            output_str = output_str + '_model'
            
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
                    
                    output_str = (output_str  + '-eig_lim-' + 
                                  str(df_data_coll_params['c_eig']))
                    
                elif df_data_coll_params['method'] == 'exclz':
                    
                    output_str = (output_str + '-r-' + 
                                  str(df_data_coll_params['c_exclz']))
                    
        if len(output_str) == 0:
                
            raise Exception('Description has not been implemented for this case.')
     
    return output_str

def load_ground_truth(path_model):
    
    # Load already existing stability data as the "ground truth" of stability.
    with open(path_model,'rb') as f:
        target_model_raw = pickle.load(f)
    
    # The saved model is GPyOpt GPModel that is a wrapper of GPy gp_regression.
    # GPyOpt is not maintained anymore, so for easier adaptation to other
    # packages, we use here GPy gp_regression model. Let's dig it out.
    target_model = target_model_raw.model
    
    
    return target_model


def bo_sim_target(bo_ground_truth_model_path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222', 
                  materials = ['CsPbI', 'MAPbI', 'FAPbI'], rounds = 10,
                  init_points = None, batch_size = 1, acquisition_function = 'EI',
                  acq_fun_params = None, df_data_coll_params = None, no_plots = False,
                  results_folder = './Results/'):

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
    
    
    stability_model = load_ground_truth(bo_ground_truth_model_path)
    
    if df_data_coll_params is not None:
        
        # Do data fusion.
        
        if df_data_coll_params['use_model']:
            
            # Fit a Gaussian process regression model to simulate the
            # data fusion variable. During the BO loop, data fusion observations
            # will be observed by sampling this model with noise.
            
            # ALL the data fusion data will be used for fitting the model.
            data_fusion_gt_model = GP_model(pd.concat(acq_fun_params['df_data']),
                                            acq_fun_params['df_target_var'],
                                            acq_fun_params['gp_lengthscale'], 
                                            acq_fun_params['gp_variance'],
                                            acq_fun_params['df_input_var'])
            
            # Then data_fusion_data is cleared. New data will be sampled only
            # when the BO algo requires it.
            acq_fun_params['df_data'] = [pd.DataFrame(columns = acq_fun_params['df_input_var'] + [acq_fun_params['df_target_var']])]
                        
            print("Data fusion model created.")
        
        if acq_fun_params['df_data'] is None:
            
            # Initialize to empty.
            acq_fun_params['df_data'] = [pd.DataFrame(columns = acq_fun_params['df_input_var'] + [acq_fun_params['df_target_var']])]
            
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
    
    composition_total = [0.995, 1] # The sum of the amount of each material will be
    # limited between these values. If you need everything to sum up to 100% within
    # 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
    # long with this value, choose [0.99,1] or even wider range.
    
    
    for k in range(rounds):
        
        # TO DO: Generalize to more than three materials or shift up to the user defined part.
        constraints[k] = [{'name': 'constr_1', 'constraint': 'x[:,0] +x[:,1] + x[:,2] - ' + str(composition_total[1])},
                           {'name': 'constr_2', 'constraint': str(composition_total[0])+'-x[:,0] -x[:,1] - x[:,2] '}
                           ]
        
    for j in range(len(materials)):
        bounds[j] = {'name': materials[j], 'type': 'continuous', 'domain': (0,1)}
    
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
    
    if df_data_coll_params is not None:
        
        data_fusion_models = [None for j in range(rounds)]
        data_fusion_data_step = [None for j in range(rounds)]
        
        
    for k in range(rounds):
        
        if df_data_coll_params is not None:
            
            if k == 0: 
                
                data_fusion_data_rounds = acq_fun_params['df_data'].copy()
                data_fusion_data_step[k] = data_fusion_data_rounds[k]
                
            else:
                
                if len(data_fusion_data_rounds) > k:
                    
                    data_fusion_data_step[k] = pd.concat([
                                    data_fusion_data_step[k-1], data_fusion_data_rounds[k]],
                                    ignore_index = True)
                    
                else:
                    
                    data_fusion_data_step[k] = data_fusion_data_step[k-1]
                
            acq_fun_params['df_data'] = data_fusion_data_step[k]
            
        ## The implementation is like this because it is compatible with IRL data fusion BO. 
        #if acquisition_function == 'EI_DFT':
        #
        #    # Collect the data fusion data observed by this round (cumulatively).
        #    if len(data_fusion_data_rounds) > k:
        #    
        #        if k == 0:
        #            data_fusion_data_step[k] = data_fusion_data_rounds[k].copy()
        #        else:
        #            data_fusion_data_step[k] = pd.concat([
        #                data_fusion_data_step[k], data_fusion_data_rounds[k]],
        #                ignore_index = True)
        
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
        
        
        # Function predictions. Each round, the BO model is trained from zero.
        # This is not computationally efficient, though. Should we modify at
        # some point? Or do we want to keep the BO model for each round intact?
        # TO DO: clean up so that works for any #D
        if function == True:
            x = df.iloc[:,0:len(materials)].values
            df['Merit'] = predict_points_noisy(stability_model, x,
                                               stability_model.Y)[0]
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
                                                normalize_Y = True,
                                                X = X_step[k],
                                                Y = Y_step[k],
                                                evaluator_type = 'local_penalization',
                                                batch_size = batch_size,
                                                acquisition_jitter = jitter,
                                                acq_fun_params = acq_fun_params
                                                )    
        #Suggest next points (samples to prepare).
        x_next[k] = BO_batch_model[k].suggest_next_locations()
        suggestion_df[k] = pd.DataFrame(x_next[k], columns = materials)
        
        # Will be used for data fusion (if gradient option is enabled) and
        # returned to the user.
        
        lengthscale_opt = BO_batch_model[k].model.model.kern.lengthscale[0]
        variance_opt = BO_batch_model[k].model.model.kern.variance[0]
        
        gradients = BO_batch_model[k].model.model.predictive_gradients(x_next[k])[0][:,:,0]
        # Maximum gradient element value to any direction of the search space for each point.
        grad_max = np.max(np.abs(gradients), axis = 1) 
        
        lengthscales.append(lengthscale_opt)
        variances.append(variance_opt)
        max_gradients.append(grad_max)
        
        if (df_data_coll_params is not None) and (len(data_fusion_data_rounds) < (k+2)):
            
            # Do data fusion. There are no existing df datapoints given by the
            # user for the next round, so suggest new ones.
            
            # Estimate if data fusion should be requested for the next round
            # suggestions.
            
            if df_data_coll_params['method'] == 'none':
                
                # 'model_none' or 'live_none'
                
                # Never sample from the model, just add an empty df for records.
                data_fusion_data_rounds.append(pd.DataFrame(columns = 
                                                            acq_fun_params['df_input_var'] + 
                                                            [acq_fun_params['df_target_var']]))
                
            
            elif df_data_coll_params['method'] == 'all':
                #acq_fun_params[2].find('_all') != -1:
                
                # Sample human always.
                
                if df_data_coll_params['use_model']:
                    
                    # Sample all the data points from the df model.
                    data_fusion_data_rounds.append(pd.DataFrame(np.concatenate((x_next[k],
                                            predict_points(data_fusion_gt_model,
                                            x_next[k])[0]), axis=1),
                        columns = acq_fun_params['df_input_var'] +
                        [acq_fun_params['df_target_var']])) # TO DO add noise
                    
                else: 
                    #acq_fun_params[2].find('model') == -1: # 'live_all'
                
                    # Request humans to give feedback from the specified samples.
                    print('Give feedback on sample quality of these samples:\n', 
                          x_next[k])
            
            elif ((df_data_coll_params['method'] == 'exclz') or 
                  (df_data_coll_params['method'] == 'eig')):
                #acq_fun_params[2].find('_necessary') != -1:
                
                # Sample only if any of the input gradients is larger than
                # kernel varience divided by its lengthscale.
                c_grad = df_data_coll_params['c_grad']
                gradient_limit = np.sqrt(variance_opt)/lengthscale_opt*c_grad
                
                #if grad_max > gradient_limit:
                #    print('Point passed the gradient limit!')
                #    #print(gradient_limit < grad_max, gradient_limit, grad_max, lengthscale_opt, variance_opt)
                
                # Pick new points for which the surrogate model has a high
                # gradient, no matter if there is an earlier data fusion point
                # nearby.
                new_df_points_x_g = x_next[k][grad_max > gradient_limit]
                #print(new_df_points_x.shape, lengthscale_opt, variance_opt, gradient_limit, grad_max) 
                
                # THIS OPTION IS NOT IN USE.
                # Pick new points for which the surrogate model does not have
                # a high gradient but they are located far away from the
                # previously sampled points (in unknown region).
                new_df_points_x_u = x_next[k][grad_max <= gradient_limit]
                    
                # Drop the points that are excluded based on existing human evals.
                if df_data_coll_params['method'] == 'exclz':
                    #(acq_fun_params[2].find('_exclz') != -1): #'_necessary_exclz'
                        
                    # If there are points to be considered based on exclusion zone.
                    if new_df_points_x_g.shape[0] > 0:
                        
                        # Drop points with an earlier data fusion point nearby.
                        # 'Nearby' is X% of the domain length here.
                        c_exclz = df_data_coll_params['c_exclz']
                        r_limit = (bounds[0]['domain'][1] - bounds[0]['domain'][0])*c_exclz/100
                        
                        index = 0
                        for l in range(len(new_df_points_x_g)): # Should be u finally!
                            
                            if data_fusion_data_step[k].shape[0] > 0:
                            
                                if np.any(np.sqrt(np.sum((
                                        data_fusion_data_step[k].iloc[:,0:len(materials)]
                                        - new_df_points_x_g[index])**2, axis=1))
                                        < r_limit):
                                    new_df_points_x_g = np.delete(new_df_points_x_g, index, axis=0) # TO DO: Test if index works correctly when batch BO is used!
                                    print('Deleted a point based on r exclusion.')
                                    
                                else:
                                    index = index + 1
     
                elif df_data_coll_params['method'] == 'eig':
                    #acq_fun_params[2].find('_eig') != -1: # 'necessary_eig'
                    
                    # If the data fusion model exists already.
                    if data_fusion_data_step[k].shape[0] > 0:
                        
                        # Drop points based on expected information gain for the human
                        # opinion model.
                        
                        # Let's re-create the human opinion model for EIG test.
                        current_df_model = GP_model(data_fusion_data_step[k],
                                                acq_fun_params['df_target_var'],
                                                acq_fun_params['gp_lengthscale'],
                                                acq_fun_params['gp_variance'],
                                                acq_fun_params['df_input_var'])
                        data_fusion_models[j] = current_df_model

                        if new_df_points_x_g.shape[0] > 0:

                            # Variance on each point x.
                            new_var_x = predict_points(current_df_model, new_df_points_x_g)[1]
                        
                            # Model variance.
                            current_var = current_df_model.Gaussian_noise.variance[0]
                            index = 0
                        
                            for l in range(len(new_df_points_x_g)): # Should be u finally!
                                
                                # EIG test:
                                #eig = 0.1 * (np.log10(new_sigma_x[0][l] + current_df_model.rbf.variance[0]) - np.log10(current_df_model.rbf.variance[0]))
                                #lim was 0.005
                                eig = 0.5 * (np.log10(new_var_x[l,0]/current_var + 1))
                                #print('EIG=', eig, ', point:', new_df_points_x_g, ', sigma^2_x=', new_var_x, ', sigma^2=', current_var)
                                
                                c_eig = df_data_coll_params['c_eig']
                                
                                # Max EIG(sigma_x=sigma) = 0.5*np.log10(2) = 0.15. Should this be some kind of reasonable limit?
                                if eig < 0.15*c_eig:
                                    
                                    new_df_points_x_g = np.delete(new_df_points_x_g, index, axis=0)
                                    print('Deleted a point based on EIG.')
                                    
                                else:
                                    
                                    index = index + 1
                        
                # NOT IN USE AT THE MOMENT.
                # Combine the two criteria.
                new_df_points_x = new_df_points_x_g # np.append(new_df_points_x_g, new_df_points_x_u, axis = 0)
                
                #new_df_points_x = new_df_points_x_g
                
                if new_df_points_x.shape[0] > 0:
                    
                    if df_data_coll_params['use_model']:
                        #acq_fun_params[2].find('_model') != -1:
                        
                       # Make predictions on the data fusion property.
                       
                       #print(new_df_points_x.shape)
                       new_df_points_y = predict_points(data_fusion_gt_model, new_df_points_x)[0] # TO DO: Add noise!
                       # TO DO check that y predictions are scaled correctly.
                       
                       # Then add to data_fusion_data
                       data_fusion_data_rounds.append(pd.DataFrame(np.concatenate((new_df_points_x, new_df_points_y),
                                      axis = 1),
                           columns = acq_fun_params['df_input_var'] +
                           [acq_fun_params['df_target_var']])) # TO DO add noise
                    
                    else:
                        
                        # Request humans to give feedback from the specified samples.
                        print('Give feedback on sample quality of these samples:\n', new_df_points_x)
                        
                    print('Round ' + str(k) + ': ', new_df_points_x)
                    
                else:
                    
                    # Add empty DataFrame
                    data_fusion_data_rounds.append(pd.DataFrame(
                        columns = acq_fun_params['df_input_var'] +
                        [acq_fun_params['df_target_var']]))
                    
    # Plot and save the results.
    
    #if function == True:
    #    degradation_input = compositions_input
    
    
    if no_plots == False:
        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
        plotBO(rounds, suggestion_df, compositions_input, degradation_input,
               BO_batch_model, materials, X_rounds, x_next, Y_step, X_step,
               limit_file_number = True, time_str = time_now,
               results_folder = results_folder)    
        
        if acquisition_function == 'EI_DFT':
            plotDF(rounds, materials, data_fusion_models, data_fusion_data_rounds, acq_fun_params['df_target_var'], 
                   acq_fun_params['gp_lengthscale'],
                   acq_fun_params['gp_variance'],
                   acq_fun_params['p_beta'], acq_fun_params['p_midpoint'],
                   limit_file_number = True, time_str = time_now,
                   results_folder = results_folder)
            
    print('Last suggestions for the next sampling points: ', x_next[-1])
    print('Results are saved into the given folder.')
    
    # Save the model as an backup
    # dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab') 
    # pickle.dump([BO_batch, suggestion_df, x_next, X_rounds, Y_rounds], dbfile)                      
    # dbfile.close()
    
    # Minimum value vs rounds.
    optimum = np.empty((rounds, len(materials) + 1))
    mod_optimum = np.empty((rounds, len(materials) + 1))
    for i in range(rounds):
        
        idx = np.argmin(Y_step[i], axis = 0)
        opt = Y_step[i][idx, 0] #np.min(Y_step[i]))
        loc = X_step[i][idx, :]
        
        optimum[i,0:len(materials)] = loc
        optimum[i,-1] = opt
        
        #mod_opt = BO_batch_model[i].model.get_fmin()
        #mod_opt, _ = BO_batch_model[i].model.predict(mod_loc)
        
        #mod_optimum[i,0:len(materials)] = mod_loc
        #mod_optimum[i,-1] = mod_opt
        
        X_rounds[i] = pd.DataFrame(data = X_rounds[i], columns = materials)
        Y_rounds[i] = pd.DataFrame(data = Y_rounds[i], columns = ['Target'])

    if no_plots == False:
            plt.figure()
            plt.plot(range(rounds), optimum)
            plt.show()

    surrogate_model_params = {'lengthscales': lengthscales,
                              'variances': variances,
                              'max_gradients': max_gradients}
    
    if df_data_coll_params is not None:
        
        data_fusion_params = {'df_data_rounds': data_fusion_data_rounds,
                              'df_data_step': data_fusion_data_step}
        
    else:
        
        data_fusion_params = None

    next_suggestions = suggestion_df.copy()
    optimum = optimum.copy()
    mod_optimum = mod_optimum.copy()
    
    X_rounds = X_rounds.copy()
    Y_rounds = Y_rounds.copy()
    
    bo_objects = BO_batch_model.copy()
    
    return next_suggestions, optimum, mod_optimum, X_rounds, Y_rounds, X_step, Y_step, surrogate_model_params, data_fusion_params, bo_objects
#optimum, suggestion_df, compositions_input, degradation_input, BO_batch_model, X_rounds, x_next, Y_step, X_step, data_fusion_data, lengthscales, variances, max_gradients
