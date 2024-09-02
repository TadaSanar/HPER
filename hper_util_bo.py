"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hper_util_gp import predict_points, predict_points_noisy

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
                
        p = {'df_kernel_lengthscale': None,
             'df_kernel_variance': None,
             'df_noise_variance': None,
             'p_beta': None,
             'p_midpoint': None,
             'df_target_variable': None, 
             'df_input_variables': None}
        
        # Check if user has provided values for any of these parameters.
        for key in p:
            
            if key in data_fusion_settings:
                
                p[key] = data_fusion_settings[key]
            
        # Else, pick the default values for the keys for each data type.
        
        if data_fusion_settings['df_target_property_name'] == 'dft':
            
            if p['df_target_variable'] == None:
                variable = 'dGmix (ev/f.u.)'
            if p['df_kernel_lengthscale'] == None:  # For Gaussian process regression
                lengthscale = 0.03
            if p['df_kernel_variance'] == None:  # For GPR
                k_variance = 2
            if p['df_noise_variance'] == None:  # For GPR
                n_variance = None
            if p['p_beta'] == None:  # For probability model
                p_beta = 0.025
            if p['p_midpoint'] == None:  # For P model
                p_midpoint = 0  # For P

        elif data_fusion_settings['df_target_property_name'] == 'quality':
            
            if p['df_target_variable'] == None:
                variable = 'Quality'
            if p['df_kernel_lengthscale'] == None:
                lengthscale = None # Set a value for the lengthscale if you have info on this, otherwise the model learns it.
            if p['df_kernel_variance'] == None:
                k_variance = 1 # Assumes the quality data is roughly zero mean unit variance.
            if p['df_noise_variance'] == None:
                n_variance = None # Set a value for the noise variance level if you have info on this, otherwise the model learns it.
            if p['p_beta'] == None:  # For probability model
                p_beta = 0.04 #0.05
            if p['p_midpoint'] == None:  # For P model
                p_midpoint = 0.33#0.5  # For P

        elif data_fusion_settings['df_target_property_name'] == 'cutoff':

            if p['df_target_variable'] == None:
                variable = 'Cutoff'
            if p['df_kernel_lengthscale'] == None:
                lengthscale = 0.05
            if p['df_kernel_variance'] == None:
                k_variance = None
            if p['df_noise_variance'] == None:
                n_variance = None
            if p['p_beta'] == None:
                p_beta = 0.025
            if p['p_midpoint'] == None:
                p_midpoint = 0

        else:
            
            raise Exception('Data fusion target variable ' + 
                            data_fusion_settings['df_target_property_name'] +
                            ' has not been implemented in the parameter ' +
                            'builder. Please provide another variable name or' +
                            'add yours into the builder.')

        ei_df_params = {'df_data': None, #data_fusion_data,
                         'df_target_var': variable,
                         'df_target_prop': data_fusion_settings['df_target_property_name'],
                         'df_input_var': p['df_input_variables'],
                         'df_kernel_lengthscale': lengthscale,
                         'df_kernel_variance': k_variance,
                         'df_noise_variance': n_variance,
                         'p_beta': p_beta,
                         'p_midpoint': p_midpoint
                         }
    
    print(ei_df_params)
    
    return ei_df_params


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
                      '-lengthscale-' + str(ei_df_params['df_kernel_lengthscale']) +
                      '-variance-' + str(ei_df_params['df_kernel_variance']) +
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


def plot_basic(Y_accum, X_accum, optimum, model_optimum, rounds, time_str = None, results_folder = None, 
               ref_x = np.array([[0.165, 0.04, 0.79]]), ref_y = 90.508, 
               ref_y_std = 20.688, saveas = False, hyperpars = None,
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

        optimum[i, 0:-1] = loc
        optimum[i, -1] = opt

    
    # Model optimum for unconstrained and constrained space.
    for i in range(rounds):
        
        model_optimum[i,0:-1], model_optimum[i,-1] = find_optimum(BO_objects[i].model.model, # GPRegression model 
                                                    Y_train = Y_accum[i],
                                                    ternary = ternary, 
                                                    domain_boundaries = 
                                                    domain_boundaries)
        
    return optimum, model_optimum






