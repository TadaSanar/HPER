#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:11:36 2024

@author: atiihone
"""

import pandas as pd
import numpy as np

from hper_util_gp import predict_points

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
        # EIG criterion.
        if new_df_points_x_g.shape[0] > 0:

            # Drop the points that are excluded from the points to be queried
            # because there are previous human evaluations nearby.
            if df_data_coll_params['method'] == 'exclz':

                # Drop points with an earlier data fusion point nearby.
                # 'Nearby' is X% of the domain length here.
                c_exclz = df_data_coll_params['c_exclz']
                r_limit = (bounds[0]['domain'][1] - bounds[0]['domain'][0])*c_exclz/100

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

                    ## Let's use the human opinion model for EIG test.
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




