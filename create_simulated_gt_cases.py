#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:54:38 2023

@author: armi
"""

#from hper_bo import plot_GP, GP_model
import pickle
from hper_bo import load_ground_truth
import GPy
from GPy.kern import Matern52, RBF
from GPy.models import GPRegression
import GPyOpt
import numpy as np

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

from plotting_v2 import triangleplot

def predict_points(gpmodel, x_points, Y_data=None):
    '''
    For a GPy GP regression model or GPyOpt GPModel.
    '''
    if type(gpmodel) is GPy.models.gp_regression.GPRegression:
        
        # Prediction output is mean, variance.
        posterior_mean, posterior_var = gpmodel.predict(x_points)
        posterior_std = np.sqrt(posterior_var)
        
    elif type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
        
        # Prediction output is mean, standard deviation.
        posterior_mean, posterior_std = gpmodel.predict(x_points)
        posterior_var = (posterior_std)**2
        
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

    # Adding Gaussian noise to the mean predictions.
    posterior_mean_noisy = np.random.normal(
        posterior_mean, np.sqrt(posterior_var)*noise_level)
    
    #logging.log(21, 'Noise level: ' + str(noise_level))
    #logging.log(21, 'Posterior mean: ' + str(posterior_mean))
    #logging.log(21, 'Posterior mean noisy: ' + str(posterior_mean_noisy))
    #logging.log(21, 'Seed: ' + str(np.random.get_state()[1][0]))
    
    return posterior_mean_noisy, posterior_var, posterior_mean

def create_ternary_grid(step = 0.005):

    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(0.0,1.0, step)
    xt, yt, zt = np.meshgrid(a,a,a, sparse=False)
    points = np.transpose([xt.ravel(), yt.ravel(), zt.ravel()])
    points = points[abs(np.sum(points, axis=1)-1) < (step - step/5)]
    
    return points

def find_minimum(model):
    
    # Assumes single-task y.
    points = create_ternary_grid()
    y, _ = model.predict(points)
    idx_min = np.argmin(y)
    y_min = y[idx_min, :]
    x_min = points[[idx_min], :]
    
    return y_min, x_min
    

def define_grid_lims_posterior(GP_model, Y_train = None, data_type = 'stability'):
    
    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_ternary_grid()
    
    # Here the posterior mean and std deviation are calculated.
    posterior_mean, posterior_var = predict_points(GP_model, points, Y_data = Y_train)
    posterior_std = np.sqrt(posterior_var)
    
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
    elif data_type == 'stability_unscaled':
        cbar_label_mean = r'$I_{c}(\theta)$ (px$\cdot$min)'
        cbar_label_std = r'Std $I_{c}(\theta)$ (px$\cdot$min)'
        saveas_mean = 'Ic-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-Ic'
        saveas_withsamples ='Modelled-Ic-with-samples'
    elif data_type == 'human':
        cbar_label_mean = r'$Human(\theta)$'
        cbar_label_std = r'Std $Human(\theta)$'
        saveas_mean = 'Human-no-grid' + np.datetime_as_string(np.datetime64('now'))
        saveas_std = 'St-Dev-of-Human'
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
              saveas = 'Triangle_surf', surf_levels = None):

    #print(y_data.shape, points.shape)
    #print(norm)
    #print(cmap)
    triangleplot(points, y_data, norm, cmap = cmap,
                 cbar_label = cbar_label, 
                 saveas = saveas, surf_levels = surf_levels)#[2e-3, 4e-3, 6e-3, 8e-3])

def plot_surf_mean(points, posterior_mean, lims, axis_scale = 1,
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)',
                   saveas = 'Ic-no-grid', surf_levels = None):
    
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])    
    y_data = posterior_mean/axis_scale
    plot_surf(points, y_data, norm, cbar_label = cbar_label, saveas = saveas,
              surf_levels = surf_levels)
    
def plot_surf_std(points, posterior_std, lims, axis_scale = 1,
                  cbar_label = r'Std of $I_{c}(\theta)$ (px$\cdot$h)',
                  saveas = 'St-Dev-of-Ic', surf_levels = None):
    
    vmax = lims[1][1]
    if vmax < 400: # Std can get very small with a dense sampling.
        vmax = 400
        
    norm = matplotlib.colors.Normalize(vmin=lims[1][0], 
                                       vmax=lims[1][1])    
    y_data = posterior_std/axis_scale
    plot_surf(points, y_data, norm, cbar_label = cbar_label, saveas = saveas, surf_levels = surf_levels)

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
            data_type = 'stability', surf_levels = None):

    points, lims, axis_scale, posterior_mean, posterior_std, cbar_labels, filenames = define_grid_lims_posterior(
        GP_model, Y_train = Y_train, data_type = data_type)
    
    #original_folder = os.getcwd()
    #os.chdir(original_folder)
    

    # Let's plot the requested points. This plot works for 3 materials only.
       
    plot_surf_mean(points, posterior_mean, lims, axis_scale = axis_scale,
                   cbar_label = cbar_labels[0], saveas = filenames[0],
                   surf_levels = surf_levels)
    plot_surf_std(points, posterior_std, lims, axis_scale = axis_scale,
                   cbar_label = cbar_labels[1], saveas = filenames[1],
                   surf_levels = surf_levels)
    
    if x_points is not None:
    	plot_surf_mean_and_points(points, posterior_mean, x_points, 
                               y_points, lims, axis_scale = axis_scale,
                               cbar_label = cbar_labels[0], saveas = filenames[2])

def predict_and_plot_points(GP_model, x_points, Y_train = None, 
                            saveas = 'Predicted-Ic-points', 
                            noisy = False, data_type = 'stability'):
    
    if noisy is False:
        y_points, y_var_points = predict_points(GP_model, x_points, Y_train = Y_train)
        y_std_points = np.sqrt(y_var_points)
        
    else:
        y_points, y_points_no_noise, y_var_points = predict_points_noisy(
            GP_model, x_points, Y_train = Y_train)
        y_std_points = np.sqrt(y_var_points)
    
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
def GP_model(files, materials = ['CsPbI', 'MAPbI', 'FAPbI'], 
             target_variable = 'dGmix (ev/f.u.)', lengthscale = 0.03, 
             variance = 2):
    
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
stability_gt_model_file = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222'
human_gt_model_file = './Source_data/visualquality/Human_GPR_model_20220801'
materials = ['CsPbI', 'MAPbI', 'FAPbI'] # Material compositions are given in this order. The current implementation may or may not work with different order, so be careful.

#########################################################################

# Load already existing stability data as the "ground truth" of stability.

gt_model = load_ground_truth(stability_gt_model_file).model
gt_lengthscale = gt_model.Mat52.lengthscale
gt_variance = gt_model.Mat52.variance
gt_noise_var = gt_model.Gaussian_noise.variance[0]

y = gt_model.Y.copy()
x = gt_model.X.copy()

print(y.min())
print(y.max())

# Plot the "ground-truth" stability model and all the data used for training it.
plot_GP(gt_model, Y_train=y, x_points = x, y_points = y, 
        data_type = 'stability_unscaled')

#########################################################################
# Retrain the model as a sanity check. Should look the same than previous plots.

#gt_model = load_ground_truth(stability_gt_model).model
x = gt_model.X.copy()
y_scaled = gt_model.Y.copy()

# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance*(y_scaled.std())**2)
noise_var = gt_noise_var * (y_scaled.std())**2
model = GPRegression(x, y_scaled, kernel, noise_var = noise_var)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

#plot_GP(model, y_scaled, x_points = x, y_points = y_scaled, 
#        data_type = 'stability')

plot_GP(model, Y_train=None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')
    
#pickle.dump(model, open('./Source_data/stability_gt_model_GPR', 'wb'))
#########################################################################
# Stability model normalized to 0 mean unit variance.

#gt_model = load_ground_truth(stability_gt_model).model
x = gt_model.X.copy()
y_scaled = (gt_model.Y.copy() - gt_model.Y.mean())/gt_model.Y.std()

# Matern kernel
kernel = Matern52(input_dim=x.shape[1])#, lengthscale=gt_lengthscale, 
#                  variance=gt_variance*(y_scaled.std())**2)
#noise_var = gt_noise_var * (y_scaled.std())**2
model = GPRegression(x, y_scaled, kernel)#, noise_var = noise_var)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())
print(y_scaled.mean(), y_scaled.std())

#plot_GP(model, y_scaled, x_points = x, y_points = y_scaled, 
#        data_type = 'stability')

plot_GP(model, Y_train=None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')
    
#pickle.dump(model, open('./Source_data/stability_model_scale1mean0std', 'wb'))
#########################################################################
# Modify by improving region B. Scaled.

x = gt_model.X.copy()
y = gt_model.Y.copy()

lim = 0.8
idx = x[:,0] > lim
scale = np.min(y)/np.min(y[idx])/2

y_scaled = y.copy()
y_scaled[idx, 0] = y_scaled[idx, 0] * scale
y_scaled = y_scaled / np.max(y_scaled)

# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance)
noise_var = gt_noise_var
model = GPRegression(x,y_scaled,kernel, noise_var = noise_var)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

plot_GP(model, Y_train = None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')

#pickle.dump(model, open('./Source_data/stability_model_improved_region_B', 'wb'))

#########################################################################
# Modify by improving region B so that it is equal to A. Uses internal
# scaling of GPy.

x = gt_model.X.copy()
y = gt_model.Y.copy()

# Improve 0% FA edge of the triangle around region B.
idx = ((x[:,0] > 0.77) & (x[:,0] < 1) & # Cs
       (x[:,1] > 0.01) & (x[:,1] < 0.1) & # MA
       (x[:,2] > -0.01) & (x[:,2] < 0.05)) # FA       
scale = np.min(y)/np.min(y[idx])

y_scaled = y.copy()
y_scaled[idx, 0] = y_scaled[idx, 0] * scale

# Create the main region B
idx = ((x[:,0] > 0.77) & (x[:,0] < 1) & # Cs
       (x[:,1] > 0.01) & (x[:,1] < 0.12) & # MA
       (x[:,2] > -0.01) & (x[:,2] < 0.15)) # FA       
scale = np.min(y)/np.min(y[idx])

y_scaled[idx, 0] = y_scaled[idx, 0] * scale

# Bend it more towards the right corner.
#idx = ((x[:,0] > 0.9) & (x[:,0] < 0.96) & # Cs
#       (x[:,1] > -0.01) & (x[:,1] < 0.15) & # MA
#       (x[:,2] > 0) & (x[:,2] < 0.05)) # FA       
#scale = np.min(y_scaled)/np.min(y_scaled[idx])
#y_scaled[idx, 0] = y_scaled[idx, 0] * scale

#y_scaled = y_scaled / np.max(y_scaled)

# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance)
noise_var = gt_noise_var
model = GPRegression(x,y_scaled,kernel, noise_var = noise_var,
                     normalizer = True)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

plot_GP(model, Y_train = None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')

#pickle.dump(model, open('./Source_data/stability_model_equal_AB', 'wb'))


#########################################################################
# Modify by making stability to look pretty much like human model (high correlation).

x = gt_model.X.copy()
y = gt_model.Y.copy()

lim = 0.8
idx = x[:,0] > lim
scale = 2.8#np.min(y)/np.min(y[idx])/2

y_scaled0 = y.copy()
y_scaled0[idx, 0] = y_scaled0[idx, 0] * scale
#y_scaled0 = y_scaled0 / np.max(y_scaled0)

lim2 = 0.24
idx2 = x[:,1] > lim2
scale2 = 0.1#np.min(y)/np.min(y[idx])/2

y_scaled = y_scaled0.copy()
y_scaled[idx2, 0] = y_scaled[idx2, 0] * scale2
#y_scaled = y_scaled / np.max(y_scaled)


# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance*(y_scaled.std())**2)
noise_var = gt_noise_var * (y_scaled.std())**2
model = GPRegression(x, y_scaled, kernel, noise_var = noise_var)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

#plot_GP(model, y_scaled, x_points = x, y_points = y_scaled, 
#        data_type = 'stability')

plot_GP(model, Y_train=None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')
    
#pickle.dump(model, open('./Source_data/stability_model_higher_correlation_with_human', 'wb'))

#########################################################################
# Modify by making stability to look pretty much like human model (high correlation).
# Plus scale to btw 0 and 3.

x = gt_model.X.copy()
y = gt_model.Y.copy()

lim = 0.8
idx = x[:,0] > lim
scale = 2.8#np.min(y)/np.min(y[idx])/2

y_scaled0 = y.copy()
y_scaled0[idx, 0] = y_scaled0[idx, 0] * scale
#y_scaled0 = y_scaled0 / np.max(y_scaled0)

lim2 = 0.24
idx2 = x[:,1] > lim2
scale2 = 0.1#np.min(y)/np.min(y[idx])/2

y_scaled = y_scaled0.copy()
y_scaled[idx2, 0] = y_scaled[idx2, 0] * scale2
y_scaled = y_scaled / np.max(y_scaled) * 3


# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance*(y_scaled.std())**2)
noise_var = gt_noise_var * (y_scaled.std())**2
model = GPRegression(x, y_scaled, kernel, noise_var = noise_var)

# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

plot_GP(model, Y_train=None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')

    
#pickle.dump(model, open('./Source_data/stability_model_higher_correlation_with_human_scale0to3', 'wb'))

#########################################################################
# Modify by making stability to look pretty much like human model (high correlation).
# Plus scale to btw 0 and 1.

x = gt_model.X.copy()
y = gt_model.Y.copy()

lim = 0.8
idx = x[:,0] > lim
scale = 2.8#np.min(y)/np.min(y[idx])/2

y_scaled0 = y.copy()
y_scaled0[idx, 0] = y_scaled0[idx, 0] * scale
#y_scaled0 = y_scaled0 / np.max(y_scaled0)

lim2 = 0.24
idx2 = x[:,1] > lim2
scale2 = 0.1#np.min(y)/np.min(y[idx])/2

y_scaled = y_scaled0.copy()
y_scaled[idx2, 0] = y_scaled[idx2, 0] * scale2
y_scaled = y_scaled / np.max(y_scaled)


# Matern kernel
kernel = Matern52(input_dim=x.shape[1], lengthscale=gt_lengthscale, 
                  variance=gt_variance*(y_scaled.std())**2)
noise_var = gt_noise_var * (y_scaled.std())**2
model = GPRegression(x, y_scaled, kernel, noise_var = noise_var)


# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

print(y.min(), y_scaled.min())
print(y.max(), y_scaled.max())

plot_GP(model, Y_train=None, x_points = x, y_points = y_scaled, 
        data_type = 'stability_unscaled')
    
#pickle.dump(model, open('./Source_data/stability_model_higher_correlation_with_human_scale0to1', 'wb'))


#########################################################################
# Human stability model.
gt_model_human = load_ground_truth(human_gt_model_file)
gt_lengthscale_human = gt_model_human.kern.lengthscale
gt_variance_human = gt_model_human.kern.variance
gt_noise_var_human = gt_model_human.Gaussian_noise.variance[0]

y_human = gt_model_human.Y.copy()
x_human = gt_model_human.X.copy()

print(y_human.min())
print(y_human.max())

plot_GP(gt_model_human, Y_train = None, x_points = x_human, y_points = y_human, 
        data_type = 'human')
    
#########################################################################
# Retrain human stability model as a sanity check. Should look the same than above.

y_human = gt_model_human.Y.copy()
x_human = gt_model_human.X.copy()

y_scaled = y_human.copy()

print(y_scaled.min())
print(y_scaled.max())

# RBF kernel
kernel = RBF(input_dim=x_human.shape[1], lengthscale=gt_lengthscale_human, 
                  variance=gt_variance_human*(y_scaled.std())**2)
noise_var = gt_noise_var_human * (y_scaled.std())**2
model = GPRegression(x_human, y_scaled, kernel, noise_var = noise_var)


# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

plot_GP(model, Y_train = None, x_points = x_human, y_points = y_scaled, 
        data_type = 'human')
    

#########################################################################
# Human stability model scaled to between 0 and 1.

y_human = gt_model_human.Y.copy()
x_human = gt_model_human.X.copy()

y_scaled = y_human.copy()/y_human.max()

print(y_scaled.min())
print(y_scaled.max())

# RBF kernel
kernel = RBF(input_dim=x.shape[1], lengthscale=gt_lengthscale_human, 
                  variance=gt_variance_human*(y_scaled.std())**2)
noise_var = gt_noise_var_human * (y_scaled.std())**2
model = GPRegression(x_human, y_scaled, kernel, noise_var = noise_var)


# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

plot_GP(model, Y_train = None, x_points = x_human, y_points = y_scaled, 
        data_type = 'human')

print(y_scaled.min())
print(y_scaled.max())
print(y_scaled.mean(), y_scaled.std())

#pickle.dump(model_human, open('./Source_data/visualquality/human_model_scale0to1', 'wb'))

#########################################################################
# Human model scaled to 0 mean unit variance.

create_ternary_grid(step = 0.005)

y_human = gt_model_human.Y.copy()
x_human = gt_model_human.X.copy()

y_scaled = (y_human.copy() - y_human.mean())/y_human.std()

# RBF kernel
kernel = RBF(input_dim=x.shape[1])#, lengthscale=gt_lengthscale_human, 
#                  variance=gt_variance_human*(y_scaled.std())**2)
#noise_var = gt_noise_var_human * (y_scaled.std())**2
model = GPRegression(x_human, y_scaled, kernel)#, noise_var = noise_var)


# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

plot_GP(model, Y_train = None, x_points = x_human, y_points = y_scaled, 
        data_type = 'human')

print(y_scaled.min())
print(y_scaled.max())
print(y_scaled.mean(), y_scaled.std())

#pickle.dump(model, open('./Source_data/visualquality/human_model_scale0mean1std', 'wb'))


#########################################################################
# Imaginary human model that resembles stability model and is scaled 
# scaled to 0 mean unit variance.

y_human = gt_model.Y.copy() # Stability data!
x_human = gt_model.X.copy()

#y_scaled = (y_human.copy() - y_human.mean())/y_human.std()
y_scaled = y_human.copy()
            
# Force region B to look bad.

lim = 0.7
idx = x[:,0] > lim
scale = 3.5#np.min(y)/np.min(y[idx])/2

y_scaled[idx, 0] = y_scaled[idx, 0] * scale
y_scaled = (y_scaled - y_scaled.mean()) / y_scaled.std()

# RBF kernel
kernel = RBF(input_dim=x.shape[1])#, lengthscale=gt_lengthscale_human, 
#                  variance=gt_variance_human*(y_scaled.std())**2)
#noise_var = gt_noise_var_human * (y_scaled.std())**2
model = GPRegression(x_human, y_scaled, kernel)#, noise_var = noise_var)


# optimize and plot
model.optimize_restarts(messages=True,max_f_eval = 100000)

plot_GP(model, Y_train = None, x_points = x_human, y_points = y_scaled, 
        data_type = 'human')

print(y_scaled.min())
print(y_scaled.max())
print(y_scaled.mean(), y_scaled.std())

#pickle.dump(model, open('./Source_data/visualquality/human_model_scale0mean1std_higher_corrwithstability', 'wb'))
