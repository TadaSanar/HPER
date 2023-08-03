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
import logging


def predict_points(GP_model, x_points, Y_data=None):
    '''
    For a GPy GP regression model.
    '''
    posterior_mean, posterior_var = GP_model.predict(x_points)

    # If the model has been trained with already-scaled (zero mean, unit
    # variance) data, the provided train data 'Y_data' will be used for scaling
    # the predictions to the correct units.
    if Y_data is not None:
        posterior_mean_true_units = posterior_mean * \
            np.std(Y_data) + np.mean(Y_data)
        posterior_var_true_units = (np.std(Y_data))**2 * posterior_var

        posterior_mean = posterior_mean_true_units
        posterior_var = posterior_var_true_units

    return posterior_mean, posterior_var


def predict_points_noisy(GP_model, x_points, Y_data=None, noise_proportion = 1):

    # Predictions.
    posterior_mean, posterior_var = predict_points(
        GP_model, x_points, Y_data=Y_data)

    # Adding Gaussian noise to the mean predictions.
    posterior_mean_noisy = np.random.normal(
        posterior_mean, np.sqrt(posterior_var)*noise_proportion)

    return posterior_mean_noisy, posterior_var, posterior_mean


def ei_dft_param_builder(acquisition_type, data_fusion_target_property='dft',
                         data_fusion_input_variables=[
                             'CsPbI', 'MAPbI', 'FAPbI'],
                         optional_acq_params=None):
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
            if p['lengthscale'] == None:  # For Gaussian process regression
                lengthscale = 0.03
            if p['variance'] == None:  # For GPR
                variance = 2
            if p['beta'] == None:  # For probability model
                beta = 0.025
            if p['midpoint'] == None:  # For P model
                midpoint = 0  # For P

        elif data_fusion_target_property == 'visual':

            if p['files'] is None:

                # Visual quality of the samples as a constraint.
                files = [
                    ['./Source_data/visualquality/visualquality_round_0-1.csv']]

            if p['df_target_variable'] == None:
                variable = 'Quality'
            if p['lengthscale'] == None:  # For GP regression
                lengthscale = 0.1
            if p['variance'] == None:  # For GPR
                variance = 0.1
            if p['beta'] == None:  # For probability model
                beta = 0.1
            if p['midpoint'] == None:  # For P model
                midpoint = 0  # For P

        elif data_fusion_target_property == 'cutoff':

            if p['files'] is None:

                # Arbitrary function (such as a direct cutoff) as a constraint.
                files = [['./arbitrary_function/cutoff.csv']]

            if p['df_target_variable'] == None:
                variable = 'Cutoff'

            if p['lengthscale'] == None:  # For Gaussian process regression
                lengthscale = 0.05
            if p['variance'] == None:  # For GPR
                variance = 0.05
            if p['beta'] == None:  # For probability model
                beta = 0.025
            if p['midpoint'] == None:  # For P model
                midpoint = 0  # For P

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
        raise Exception(
            'The parameter builder is not required for this acquisition function. The builder has been implemented only for EI_DFT.')

    return ei_dft_params


def acq_param_builder(acquisition_function, data_fusion_property=None,
                      data_fusion_input_variables=None, optional_acq_params=None):

    if data_fusion_property is None:

        acq_fun_params = {}

    else:

        if acquisition_function == 'EI_DFT':

            acq_fun_params = ei_dft_param_builder(acquisition_function,
                                                  data_fusion_target_property=data_fusion_property,
                                                  data_fusion_input_variables=data_fusion_input_variables,
                                                  optional_acq_params=optional_acq_params)

        else:

            raise Exception(
                'Data fusion has not been implemented for this acquisition function.')
    
    acq_fun_params['jitter'] = 0.01
    
    if optional_acq_params is not None:
        
        if 'jitter' in optional_acq_params:
            
            acq_fun_params['jitter'] = optional_acq_params['jitter']
    
    return acq_fun_params


def acq_fun_param2descr(acq_fun, acq_fun_params=None):

    output_str = acq_fun

    if acq_fun == 'EI_DFT':

        ei_dft_params = acq_fun_params
        output_str = (output_str + '-dftarget-' + ei_dft_params['df_target_prop'] +
                      '-lengthscale-' + str(ei_dft_params['gp_lengthscale']) +
                      '-variance-' + str(ei_dft_params['gp_variance']) +
                      '-beta-' + str(ei_dft_params['p_beta']) +
                      '-midpoint-' + str(ei_dft_params['p_midpoint']))
        
    if acq_fun_params is not None:
        
        if 'jitter' in acq_fun_params:
            
            output_str = output_str + '-jitter-' + str(acq_fun_params['jitter'])
        
    return output_str


def df_data_coll_param_builder(df_method=None, gradient_param=None, exclusion_param=None):

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

            raise Exception(
                'Data fusion data collection has not been implemented for this case.')

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

                    output_str = (output_str + '-eig_lim-' +
                                  str(df_data_coll_params['c_eig']))

                elif df_data_coll_params['method'] == 'exclz':

                    output_str = (output_str + '-r-' +
                                  str(df_data_coll_params['c_exclz']))

        if len(output_str) == 0:

            raise Exception(
                'Description has not been implemented for this case.')

    return output_str


def build_constraint_str(materials, composition_total, prefix='x[:,',
                         postfix=']'):

    c1 = ''
    c0 = str(composition_total[0])

    for idx in range(len(materials)):
        c1 = c1 + prefix + str(idx) + postfix + ' + '
        c0 = c0 + ' - ' + prefix + str(idx) + postfix

    c1 = c1[0:-2] + '- ' + str(composition_total[1])

    return c0, c1


def load_ground_truth(path_model):

    # Load already existing stability data as the "ground truth" of stability.
    with open(path_model, 'rb') as f:
        target_model_raw = pickle.load(f)

    # The saved model is GPyOpt GPModel that is a wrapper of GPy gp_regression.
    # GPyOpt is not maintained anymore, so for easier adaptation to other
    # packages, we use here GPy gp_regression model. Let's dig it out.
    target_model = target_model_raw.model

    return target_model

def query_data_fusion_data_from_model(k, data_fusion_XZ_rounds, 
                                      data_fusion_XZ_accum, init_points, 
                                      data_fusion_x_next, data_fusion_gt_model, 
                                      rounds, materials, 
                                      acq_fun_params, noise = False):

    # Data fusion data is generated using the ground truth
    # model.

    if k == 0:
        
        # Sample the initial points given by the user.
        
        # Predict.
        
        # Note: There's no need to add train data to this GP model
        # because it has not been trained with scaled Y data (unlike
        # the target data model).

        if noise == False:

            data_fusion_z_next = predict_points(
                data_fusion_gt_model, np.array(init_points))[0]
        else:

            data_fusion_z_next = predict_points_noisy(
                data_fusion_gt_model, np.array(init_points))[0]

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
            if noise == False:

                data_fusion_z_next = predict_points(
                    data_fusion_gt_model, data_fusion_x_next[k-1].values)[0]
            else:

                data_fusion_z_next = predict_points_noisy(
                    data_fusion_gt_model, data_fusion_x_next[k-1].values)[0]

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
                                 init_points, x_next, stability_model, rounds,
                                 materials, noise=True):

    if (k == 0):

        # Initialization with the given grid points.
        X_rounds[k] = pd.DataFrame(init_points, columns=materials)
        
    else:

        # The locations suggested after the previous BO round will be
        # sampled in this round.
        X_rounds[k] = pd.DataFrame(x_next[k-1], columns=materials)

        #compositions_input.append(pd.DataFrame(x_next[k-1], columns = materials))

        #df = compositions_input[k].copy()

    # Function predictions. Each round, the BO model is trained from zero.
    # This is not computationally efficient, though. Should we modify at
    # some point? Or do we want to keep the BO model for each round intact?
    # TO DO: clean up so that works for any #D

    #x = df.iloc[:,0:len(materials)].values

    if noise == True:

        preds = predict_points_noisy(stability_model, X_rounds[k].values,
                                     stability_model.Y)[0]

    else:

        preds = predict_points(stability_model, X_rounds[k].values,
                               stability_model.Y)[0]

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
        gradient_limit = (np.sqrt(variance_s)/lengthscale_s)*c_grad

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
                            logging.info(message)

                        else:

                            index = index + 1

            elif df_data_coll_params['method'] == 'eig':

                # If the data fusion model exists already.
                if data_fusion_XZ_accum[k].shape[0] > 0:

                    # Drop points if the expected information gain for the
                    # human opinion model is too low.

                    # Let's re-create the human opinion model for EIG test.
                    current_df_model = GP_model(data_fusion_XZ_accum[k],
                                                acq_fun_params['df_target_var'],
                                                acq_fun_params['gp_lengthscale'],
                                                acq_fun_params['gp_variance'],
                                                acq_fun_params['df_input_var'])
                    
                    # Variance on each point x (pred. from the data fusion
                    # model).
                    var_d_next = predict_points(
                        current_df_model, new_df_points_x_g)[1]

                    # Data fusion model y variance estimate.
                    vary_d = current_df_model.Gaussian_noise.variance[0]
                    message = 'Data fusion Gaussian noise variance: ' + str(vary_d)
                    logging.log(21, message)

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
                            logging.log(21, message)

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


def bo_sim_target(bo_ground_truth_model_path='./Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222',
                  materials=['CsPbI', 'MAPbI', 'FAPbI'], rounds=10,
                  init_points=None, batch_size=1,
                  acquisition_function='EI', acq_fun_params=None,
                  df_data_coll_params=None, no_plots=False,
                  results_folder='./Results/', noise_df = False, 
                  noise_target = False):
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
    jitter = 0.01  # The level of exploration.

    # No real experiments but data is queried online from a function.
    simulated_bo = True

    ###########################################################################
    # SET DATA FUSION PARAMETERS

    if simulated_bo == True:

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

            # Then data_fusion_data is cleared now that it has been used for
            # fitting the model. New data will be sampled only when the BO algo
            # requires it.
            acq_fun_params['df_data'] = [pd.DataFrame(columns=acq_fun_params['df_input_var'] +
                                                      [acq_fun_params['df_target_var']
                                                       ])]
            
        if acq_fun_params['df_data'] is None:

            # Initialize to empty.
            acq_fun_params['df_data'] = [pd.DataFrame(columns=acq_fun_params['df_input_var'] +
                                                      [acq_fun_params['df_target_var']
                                                       ])]

        # Else, the data fusion data is already in acq_fun_params['df_data'] in
        # the required format (list of DataFrames for each round).

    ###########################################################################
    # INITIALIZE VARIABLES.

    # Material composition needs to sum up to 1 within the accuracy defined in
    # 'composition_total'.
    c0, c1 = build_constraint_str(materials, composition_total)

    # Search constraints.
    constraints = []
    for k in range(rounds):

        constraints.append([{'name': 'constr_1', 'constraint': c0},
                            {'name': 'constr_2', 'constraint': c1}])

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
        # in question).
        data_fusion_models = [None for j in range(rounds)]

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
                stability_model, rounds, materials, noise = noise_target)
            
            if df_data_coll_params is not None:
                
                # Do data fusion.
                
                data_fusion_XZ_rounds, data_fusion_XZ_accum = query_data_fusion_data_from_model(
                    k, data_fusion_XZ_rounds, data_fusion_XZ_accum, init_points,
                    data_fusion_x_next, data_fusion_gt_model, rounds, materials, 
                    acq_fun_params, noise = noise_df)
            
                # Save the data fusion data for this round to the params that will
                # be sent to the BO.
                acq_fun_params['df_data'] = data_fusion_XZ_accum[k]
                
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
                                                            evaluator_type='local_penalization',
                                                            batch_size=batch_size,
                                                            acquisition_jitter=acq_fun_params['jitter'],
                                                            acq_fun_params=acq_fun_params,
                                                            noise_var = 0.1*(Y_accum[k]/Y_accum[k].max()).var(), # GPyOpt assumes normalized Y data at the point when variance is defined.
                                                            optimize_restarts = 10,
                                                            max_iters = 2000,
                                                            exact_feval = (not noise_target)
                                                            )

        # Suggest next points (samples to prepare).
        x_next[k] = BO_objects[k].suggest_next_locations()
        # Same as a list of DataFrames for the convenience of the user.
        x_next_df[k] = pd.DataFrame(x_next[k], columns=materials)

        # Fetch surrogate model parameters to be forwarded for the user and, if
        # data fusion is enabled, used in evaluating if data fusion data should
        # be queried.
        lengthscales[k] = BO_objects[k].model.model.kern.lengthscale[0]
        variances[k] = BO_objects[k].model.model.kern.variance[0]
        gradients = BO_objects[k].model.model.predictive_gradients(x_next[k])[
            0][:, :, 0]
        gaussian_noises[k] = BO_objects[k].model.model.Gaussian_noise.variance[0]

        # Maximum gradient element value to any direction of the search space
        # for each x_next point.
        max_gradients[k] = np.max(np.abs(gradients), axis=1)

        current_surrogate_model_params = {'lengthscale': lengthscales[k],
                                          'variance': variances[k],
                                          'max_gradient': max_gradients[k]}
        
        if df_data_coll_params is not None:

            # Do data fusion.

            # Estimate if data fusion should be requested for (some of) the
            # suggestions for the next round.
            data_fusion_x_next[k] = determine_data_fusion_points(
                data_fusion_XZ_accum, df_data_coll_params, acq_fun_params,
                x_next, current_surrogate_model_params, materials, bounds, k)

    ###########################################################################
    # DATA TREATMENT, PLOTTING, SAVING

    message = 'Last suggestions for the next sampling points: ' + str(x_next[-1])
    logging.info(message)
    
    # Save the model as an backup
    # dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab')
    # pickle.dump([BO_batch, x_next_df, x_next, X_rounds, Y_rounds], dbfile)
    # dbfile.close()

    # Minimum value vs rounds (from the samples).
    optimum = np.full((rounds, len(materials) + 1), np.nan)
    # Minimum value vs rounds (from the model).
    #mod_optimum = np.full((rounds, len(materials) + 1), np.nan)

    for i in range(rounds):

        idx = np.argmin(Y_accum[i], axis=0)
        opt = Y_accum[i][idx, 0]
        loc = X_accum[i][idx, :]

        optimum[i, 0:len(materials)] = loc
        optimum[i, -1] = opt

    if no_plots == False:

        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())

        plotBO(rounds, x_next_df, BO_objects, materials, X_rounds, Y_rounds,
               Y_accum, X_accum, x_next, limit_file_number=True,
               time_str=time_now, results_folder=results_folder)

        if acquisition_function == 'EI_DFT':

            plotDF(rounds, materials, data_fusion_models,
                   data_fusion_XZ_rounds, acq_fun_params['df_target_var'],
                   acq_fun_params['gp_lengthscale'],
                   acq_fun_params['gp_variance'],
                   acq_fun_params['p_beta'], acq_fun_params['p_midpoint'],
                   limit_file_number=True, time_str=time_now,
                   results_folder=results_folder)
        
        message = 'Results are saved into the given folder.'
        logging.info(message)

        plt.figure()
        plt.plot(range(rounds), optimum)
        plt.show()

    message = ('Gaussian noises in this run: ' + str(gaussian_noises) + '\n' +
               'Lenghthscales in this run: ' + str(lengthscales) + '\n' +
               'Variances in this run: ' + str(variances)  + '\n' +
               'Max gradients in this run: ' + str(max_gradients) + '\n' +
               'Results are saved into the given folder.')
    logging.info(message)

    surrogate_model_params = {'lengthscales': lengthscales,
                              'variances': variances,
                              'max_gradients': max_gradients,
                              'gaussian_noises': gaussian_noises}

    if df_data_coll_params is not None:

        data_fusion_params = {'df_data_rounds': data_fusion_XZ_rounds,
                              'df_data_accum': data_fusion_XZ_accum}

    else:

        data_fusion_params = None

    # Not sure if these are needed anymore. I used to have some memory issues
    # that seemed to have gotten fixed by adding these, did not debug at the
    # time.
    next_suggestions = x_next_df.copy()
    optimum = optimum.copy()
    mod_optimum = None  # mod_optimum.copy()
    X_rounds = X_rounds.copy()
    Y_rounds = Y_rounds.copy()
    BO_objects = BO_objects.copy()

    return next_suggestions, optimum, mod_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, BO_objects
