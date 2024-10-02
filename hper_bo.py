"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import psutil
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Key data fusion functionalities
from hper_fun import determine_data_fusion_points

# Helper functions that are not specific to GPyOpt or GPy Bayesian optimization and Gaussian process regression packages.
from hper_util_bo import fill_accum_df_with_this_round, query_target_data_from_model, query_data_fusion_data_from_model, create_optima_arrays, plot_basic
from hper_plots_target import plotBO
from hper_plots_data_fusion import plotDF

# Helper functions that need to be modified if you switch away from GPy + GPyOpt.
from hper_util_bo_gpyopt import build_constraint_str, run_bo
from hper_util_gp import GP_model, evaluate_GP_model_constraints, constrain_optimize_GP_model, extract_gpmodel_params


def bo_sim_target(targetprop_data_source,
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
    composition_total = [0.99, 1] #Ennen [0.995,1]
    # limited between these values. If you need everything to sum up to 100% within
    # 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
    # long with this value, choose [0.99,1] or even wider range.

    # Input parameter values are within this range (all the dimensions).
    domain_boundaries = [0, 1]

    # These variables are related to the Bayesian optimization.
    num_cores = 1  # Not a parallel run
    
    if type(targetprop_data_source) is list:
        
        # List of experimental data dataframes. A feature that has not been tested fully.
        simulated_bo = False
        
    else:
        
        ## No real experiments but data is queried online from a function.
        simulated_bo = True
        

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

        if df_data_coll_params['use_model'] is False:
            
            # List of real human observations in dataframes. A feature that has not been tested fully.
            data_datafusionprop = human_data_source
            
        else:
            
            ## No real experiments but data is queried online from a function.
            gt_model_datafusionprop = human_data_source
               
            # Then data_fusion_data is cleared now that it has been used for
            # fitting the model. New data will be sampled only when the BO algo
            # requires it.
            acq_fun_params['df_data'] = [pd.DataFrame(
                columns=acq_fun_params['df_input_var'] + 
                [acq_fun_params['df_target_var']])]
                
        
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
                                            lengthscale = acq_fun_params['df_kernel_lengthscale'],
                                            variance = acq_fun_params['df_kernel_variance'],
                                            noise_variance = acq_fun_params['df_noise_variance'],
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
                        noise_variance = acq_fun_params['df_noise_variance'], 
                        kernel_variance = acq_fun_params['df_kernel_variance'], 
                        lengthscale = acq_fun_params['df_kernel_lengthscale'],
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
        BO_objects[k], x_next[k], current_surrogate_model_params = run_bo(
            X= X_accum[k], Y = Y_accum[k], bounds = bounds, 
            constraints = constraints[k], 
            acquisition_function = acquisition_function, 
            acq_fun_params = acq_fun_params, batch_size = batch_size, 
            exact_feval = exact_feval)
        
        # The next points to sample as a list of DataFrames for the convenience
        # of the user.
        x_next_df[k] = pd.DataFrame(x_next[k], columns=materials)
        
        lengthscales[k] = current_surrogate_model_params['lengthscale']
        variances[k] = current_surrogate_model_params['variance']
        gradients = current_surrogate_model_params['gradients']
        gaussian_noises[k] = current_surrogate_model_params['gaussian_noise']
        max_gradients[k] = current_surrogate_model_params['max_gradient']
        
        if df_data_coll_params is not None:
            
            # Do data fusion.
            
            # Estimate if data fusion should be requested for (some of) the
            # suggestions for the next round.
            data_fusion_x_next[k] = determine_data_fusion_points(
                data_fusion_XZ_accum, df_data_coll_params, acq_fun_params,
                x_next, current_surrogate_model_params, materials, bounds, k)
            
            data_fusion_lengthscales[k], data_fusion_variances[k], data_fusion_gaussian_noises[k] = extract_gpmodel_params(current_df_model)
            
    
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

        if (acquisition_function == 'EI_DF') or (acquisition_function == 'LCB_DF'):

            # Plot ternary-specific plots regarding data fusion.
            plotDF(rounds, materials, data_fusion_models,
                   data_fusion_XZ_accum, acq_fun_params['df_target_var'],
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
    #print(message)
                                                        
    # Not sure if these are needed anymore. I used to have some memory issues
    # that seemed to have gotten fixed by adding these, did not debug at the
    # time.
    next_suggestions = x_next_df.copy()
    optimum = optimum.copy()
    X_rounds = X_rounds.copy()
    Y_rounds = Y_rounds.copy()
    
    print('After plots and saves:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    plt.close()
    
    print('After closing figs:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    
    if (save_memory is True):
        
        BO_objects = [None] * (len(BO_objects))
        
        if (acquisition_function == 'EI_DF') or (acquisition_function == 'LCB_DF'):
            
            data_fusion_models = [None] * (len(data_fusion_models))
        
    print('After clear-outs:\n')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    
    
    return next_suggestions, optimum, model_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, BO_objects
