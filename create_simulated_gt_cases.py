#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:54:38 2023

@author: armi
"""

#from hper_bo import plot_GP, GP_model
import pickle
from hper_bo import load_ground_truth
from GPy.kern import Matern52
from GPy.models import GPRegression
import numpy as np

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

from plotting_v2 import triangleplot

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

    # Mean predictions.
    posterior_mean, posterior_std = predict_points(GP_model, x_points, Y_train = Y_train)
    
    # Adding noise to the predictions.
    noise = np.random.rand(posterior_mean.shape[0], posterior_mean.shape[1]) # Values between 0 and 1.
    noise_scaled = (noise - 0.5)*2.0
    
    posterior_mean_noisy = posterior_mean + noise_scaled*posterior_std
    
    return posterior_mean_noisy, posterior_mean, posterior_std    

def create_ternary_grid():

    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(0.0,1.0, 0.005)
    xt, yt, zt = np.meshgrid(a,a,a, sparse=False)
    points = np.transpose([xt.ravel(), yt.ravel(), zt.ravel()])
    points = points[abs(np.sum(points, axis=1)-1)<0.005]
    
    return points

def define_grid_lims_posterior(GP_model, Y_train = None, data_type = 'stability'):
    
    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_ternary_grid()
    
    # Here the posterior mean and std deviation are calculated.
    posterior_mean, posterior_std = predict_points(GP_model, points, Y_train = Y_train)
        
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    if data_type == 'stability':
    	axis_scale = 60 # Units from px*min to px*hour.
    else:
    	axis_scale = 1
    lims = [[np.min(posterior_mean)/axis_scale, np.max(posterior_mean)/axis_scale],
            [np.min(posterior_std)/axis_scale, np.max(posterior_std)/axis_scale]] # For mean and std.

    if data_type == 'stability':
        cbar_label_mean = r'$I_{c}(\theta)$ (px$\cdot$h)'
        cbar_label_std = r'Std $I_{c}(\theta)$ (px$\cdot$h)'
        saveas_mean = 'Ic-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-Ic'
        saveas_withsamples ='Modelled-Ic-with-samples'
    elif data_type == 'dft':
        cbar_label_mean = r'$dG_{mix}$ (eV/f.u.)'
        cbar_label_std = r'Std $dG_{mix}$ (eV/f.u.)'
        saveas_mean = 'dGmix-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-dGmix'
        saveas_withsamples ='Modelled-dGmix-with-samples'
    elif data_type == 'uniformity':
        cbar_label_mean = r'Uniformity\n(0=high, 3=low)'
        cbar_label_std = r'Std uniformity\n(0=high, 3=low)'
        saveas_mean = 'Uniformity-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-uniformity'
        saveas_withsamples ='Modelled-uniformity-with-samples'
    elif data_type == 'yellowness':
        cbar_label_mean = 'Yellowness (0=high, 3=low)'
        cbar_label_std = r'Std of yellowness\n(0=high, 3=low)'
        saveas_mean = 'Yellowness-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-yellowness'
        saveas_withsamples ='Modelled-yellowness-with-samples'
    else:
        cbar_label_mean = r''
        cbar_label_std = r'Std'
        saveas_mean = 'Unknown-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-unknown'
        saveas_withsamples ='Modelled-unknown-with-samples'
    cbar_labels = [cbar_label_mean, cbar_label_std]
    filenames = [saveas_mean, saveas_std, saveas_withsamples]


    return points, lims, axis_scale, posterior_mean, posterior_std, cbar_labels, filenames

def plot_surf(points, y_data, norm, cmap = 'RdBu_r', cbar_label = '',
              saveas = 'Triangle_surf'):

    #print(y_data.shape, points.shape)
    #print(norm)
    #print(cmap)
    triangleplot(points, y_data, norm, cmap = cmap,
                 cbar_label = 'I$_c$($\Theta$) (a.u.)',#cbar_label, 
                 saveas = saveas)#, surf_levels = [2e-3, 4e-3, 6e-3, 8e-3])

def plot_surf_mean(points, posterior_mean, lims, axis_scale = 1,
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)',
                   saveas = 'Ic-no-grid'):
    
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])    
    y_data = posterior_mean/axis_scale
    plot_surf(points, y_data, norm, cbar_label = cbar_label, saveas = saveas)
    
def plot_surf_std(points, posterior_std, lims, axis_scale = 1,
                  cbar_label = r'Std $I_{c}(\theta)$ (px$\cdot$h)',
                  saveas = 'St-Dev-of-Ic'):
    
    norm = matplotlib.colors.Normalize(vmin=lims[1][0], 
                                       vmax=400)#lims[1][1])    
    y_data = posterior_std/axis_scale
    plot_surf(points, y_data, norm, cbar_label = cbar_label, saveas = saveas)

def plot_surf_mean_and_points(grid_points, grid_posterior_mean, x_points, 
                              y_points, lims, axis_scale = 1, 
                              cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', 
                              saveas = 'Modelled-Ic-with-samples'):

    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])
    
    triangleplot(grid_points, grid_posterior_mean/axis_scale, norm,
	        cbar_label = cbar_label,
	        saveas = saveas,
	        scatter_points=x_points,
	        scatter_color = np.ravel(y_points/axis_scale),
	        cbar_spacing = None, cbar_ticks = None)	        

def plot_GP(GP_model, Y_train = None, x_points = None, y_points = None, 
            data_type = 'stability'):

    points, lims, axis_scale, posterior_mean, posterior_std, cbar_labels, filenames = define_grid_lims_posterior(
        GP_model, Y_train = Y_train, data_type = data_type)
    
    #original_folder = os.getcwd()
    #os.chdir(original_folder)
    

    # Let's plot the requested points. This plot works for 3 materials only.
       
    plot_surf_mean(points, posterior_mean, lims, axis_scale = axis_scale,
                   cbar_label = cbar_labels[0], saveas = filenames[0])
    plot_surf_std(points, posterior_std, lims, axis_scale = axis_scale,
                   cbar_label = cbar_labels[1], saveas = filenames[1])
    
    if x_points is not None:
    	plot_surf_mean_and_points(points, posterior_mean, x_points, 
                               y_points, lims, axis_scale = axis_scale,
                               cbar_label = cbar_labels[0], saveas = filenames[2])

def predict_and_plot_points(GP_model, x_points, Y_train = None, 
                            saveas = 'Predicted-Ic-points', 
                            noisy = False, data_type = 'stability'):
    
    if noisy is False:
        y_points, y_std_points = predict_points(GP_model, x_points, Y_train = Y_train)
    else:
        y_points, y_points_no_noise, y_std_points = predict_points_noisy(
            GP_model, x_points, Y_train = Y_train)
    
    points, lims, axis_scale, grid_posterior_mean, grid_posterior_std, cbar_labels, filenames = define_grid_lims_posterior(
        GP_model, Y_train = Y_train, data_type = data_type)
    
    if x_points is not None:
    	plot_surf_mean_and_points(points, grid_posterior_mean, x_points, 
                               y_points, lims, axis_scale = axis_scale,
                               cbar_label = cbar_labels[0],
                               saveas = filenames[2])
    	
    return y_points, y_std_points

def plot_P(GP_model, beta = 0.025, data_type = 'dft', midpoint = 0):
    
    points = create_ternary_grid()
    lims = [[0,1], [0,1]] # For mean and std. Std lims are not actually used for P.
    
    if data_type == 'stability':
        cbar_label_mean = r'$P_{Ic}$'
        saveas_mean = 'P-Ic-no-grid'
    elif data_type == 'dft':
        cbar_label_mean = r'$P_{phasestable}$'
        saveas_mean = 'P-dGmix-no-grid'
    elif data_type == 'uniformity':
        cbar_label_mean = r'P_{uniform}'
        saveas_mean = 'P-Uniformity-no-grid'
    elif data_type == 'yellowness':
        cbar_label_mean = r'$P_{dark}$'
        saveas_mean = 'P-Yellowness-no-grid'
    else:
        cbar_label_mean = r'P'
        saveas_mean = 'P-no-grid'

    mean, propability, conf_interval = calc_P(GP_model, points, beta = beta, midpoint = midpoint)
    
    plot_surf_mean(points, propability, lims, axis_scale = 1,
                   cbar_label = cbar_label_mean, saveas = saveas_mean)
    
    minP = np.min(propability)
    maxP = np.max(propability)
    
    return minP, maxP

# Added the rest of the file on 2021/11/02.
def GP_model(files, materials = ['CsPbI', 'MAPbI', 'FAPbI'], target_variable = 'dGmix (ev/f.u.)', lengthscale = 0.03, variance = 2):
    
    input_data = []
    for i in range(len(files)):
        input_data.append(pd.read_csv(files[i]))
    input_data = pd.concat(input_data)

    X = input_data[materials] # This is 3D input
    Y = input_data[[target_variable]] # Negative value: stable phase. Uncertainty = 0.025 
    X = X.iloc[:,:].values # Optimization did not succeed without type conversion.
    Y = Y.iloc[:,:].values
    # RBF kernel
    kernel = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
    model = GPy.models.GPRegression(X,Y,kernel)
    
    # optimize and plot
    model.optimize(messages=True,max_f_eval = 1000)
    
    
    return model
    
def calc_P(GP_model, points, beta = 0.025, midpoint = 0):
    
    mean = GP_model.predict_noiseless(points)
    mean = mean[0] # TO DO: issue here with dimensions?
    conf_interval = GP_model.predict_quantiles(np.array(points)) # 95% confidence interval by default. TO DO: Do we want to use this for something?

    propability = 1/(1+np.exp((mean-midpoint)/beta)) # Inverted because the negative Gibbs energies are the ones that are stable.
    
    return mean, propability, conf_interval

# TO DO: Is this function deprecated or still in use somewhere? The above is almost the same.
def mean_and_propability(x, model):#, variables):
    mean = model.predict_noiseless(x) # Manual: "This is most likely what you want to use for your predictions."
    mean = mean[0] # TO DO: issue here with dimensions?
    conf_interval = model.predict_quantiles(np.array(x)) # 95% confidence interval by default. TO DO: Do we want to use this for something?

    propability = 1/(1+np.exp(mean/0.025)) # Inverted because the negative Gibbs energies are the ones that are stable.
    
    return mean, propability, conf_interval

#########################################################################
# INPUTS
stability_gt_model = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222'

materials = ['CsPbI', 'MAPbI', 'FAPbI'] # Material compositions are given in this order. The current implementation may or may not work with different order, so be careful.

#########################################################################

# Load already existing stability data as the "ground truth" of stability.

gt_model = load_ground_truth(stability_gt_model)#.model

y = gt_model.Y
x = gt_model.X

# Plot the "ground-truth" stability model and all the data used for training it.
plot_GP(gt_model, y, x_points = x, y_points = y, 
        data_type = 'stability')


#########################################################################
# Modify by improving region B.
lim = 0.8
idx = x[:,0] > lim
scale = np.min(y)/np.min(y[idx])/2

y_scaled = y.copy()
y_scaled[idx, 0] = y_scaled[idx, 0] * scale
y_scaled = y_scaled / np.max(y_scaled)

# RBF kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_model.Mat52.lengthscale, variance=gt_model.Mat52.variance)
model = GPRegression(x,y_scaled,kernel)

# optimize and plot
model.optimize(messages=True,max_f_eval = 10000)

plot_GP(model, y_scaled, x_points = x, y_points = y_scaled, 
        data_type = 'stability')

#pickle.dump(model, open('./Source_data/stability_model_improved_region_B', 'wb'))
