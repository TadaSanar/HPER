import pickle
import GPyOpt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plotting_v2 import triangleplot

import GPy

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

    print(y_data.shape, points.shape)
    print(norm)
    print(cmap)
    triangleplot(points, y_data, norm, cmap = cmap,
                 cbar_label = cbar_label, saveas = saveas)

def plot_surf_mean(points, posterior_mean, lims, axis_scale = 1,
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)',
                   saveas = 'Ic-no-grid'):
    
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])    
    y_data = posterior_mean/axis_scale
    plot_surf(points, y_data, norm, cbar_label = cbar_label, saveas = saveas)
    
def plot_surf_std(points, posterior_std, lims, axis_scale = 1,
                  cbar_label = r'Std $I_{c}(\theta)$ (px$\cdot$h)',
                  saveas = 'St-Dev-of-Ic'):
    
    norm = matplotlib.colors.Normalize(vmin=lims[1][0], vmax=lims[1][1])    
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
#########################################################################
# INPUTS

stability_exp_results_path = './Source_data/Backup-model-20190730172222' # This pickle file contains all the results from the C2a optimization.
yellowness_model_path = './Source_data/Yellowness_GPR_model_20220801'

new_points = [[0.4,0.25,0.35]] # Give a list of points, each point is a list of 3 dimensions. These points will be predicted.
materials = ['CsPbI', 'MAPbI', 'FAPbI'] # Material compositions are given in this order. The current implementation may or may not work with different order, so be careful.



#########################################################################

# Load already existing stability data as the "ground truth" of stability.

with open(stability_exp_results_path,'rb') as f:
    [BO_batch_objects, next_sugg_df, x_next_sugg, X_rounds, Y_rounds] = pickle.load(f)

stability_model = BO_batch_objects[-1].model

x_stability = np.concatenate(X_rounds)
y_stability = np.concatenate(Y_rounds)

# Plot the "ground-truth" stability model and all the data used for training it.
plot_GP(stability_model, y_stability, x_points = x_stability, y_points = y_stability, 
        data_type = 'stability')

# Predict new point.
new_points_np = np.array(new_points)

predicted_y_points, predicted_y_std_points = predict_points(stability_model, 
                                                            new_points_np, 
                                                            y_stability)

print('Predicted values:\n', predicted_y_points/60.0, '\nAnd their st devs:\n',
      predicted_y_std_points/60.0)

# Or predict new point and plot it to the triangle.
predicted_y_points, predicted_y_std_points = predict_and_plot_points(stability_model, 
                                                                     new_points_np, 
                                                                     y_stability)

print('Predicted values:\n', predicted_y_points/60.0, '\nAnd their st devs:\n',
      predicted_y_std_points/60.0)

# Or predict a new point and add random variation with the scale of GP model 
#std to it, in order to model sample variability.
predicted_y_points, predicted_y_points_no_noise, predicted_y_std_points = predict_points_noisy(
    stability_model, new_points_np, y_stability)

print('Predicted values with noise added:\n', predicted_y_points/60.0, '\nAnd their st devs:\n',
      predicted_y_std_points/60.0)

predicted_y_points, predicted_y_std_points = predict_and_plot_points(
    stability_model, new_points_np, y_stability, noisy = True)

print('Predicted values with noise added:\n', predicted_y_points/60.0, '\nAnd their st devs:\n',
      predicted_y_std_points/60.0)

##########################################################################

# Create the visual data GP model. Either as a standalone or as DFT model.
# Optimize hyperparams with BO? Use similar file format than here:
'''
# These files contain DFT data that is integrated into the optimization loop as
# a soft constraint.
DFT_files = ['/phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
             '/phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
             '/phasestability/CsMA/fulldata/CsMA_T300_above.csv'
             ]
for n in range(len(DFT_files)):
    DFT_files[n] = original_folder + DFT_files[n]
'''






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
    model.optimize(messages=True,max_f_eval = 100)
    
    
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

'''
# For testing of GP_model() and mean_and_propability() with DFT data:
DFT_files = ['./phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
             './phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
             './phasestability/CsMA/fulldata/CsMA_T300_above.csv']

DFT_model = GP_model(DFT_files)

plot_GP(DFT_model, x_points = DFT_model.X, y_points = DFT_model.Y, 
             data_type = 'dft')

predicted_y_points, predicted_y_std_points = predict_and_plot_points(
    DFT_model, new_points_np, data_type = 'dft')

print('Predicted DFT values:\n', predicted_y_points, '\nAnd their st devs:\n',
      predicted_y_std_points)

plot_P(DFT_model, beta = 0.025, data_type = 'dft')
'''
###############################################################################

'''
# For doing the same with a model estimating the spatial uniformity of the sample films.

visual_files = ['./visualquality/visualquality_round_0-1.csv']

uniformity_model = GP_model(visual_files, materials = materials, target_variable = 'Uniformity', lengthscale = 0.1, variance = 0.1)

plot_GP(uniformity_model, x_points = uniformity_model.X, y_points = uniformity_model.Y, 
             data_type = 'uniformity')

minP, maxP = plot_P(uniformity_model, beta = 0.5, data_type = 'uniformity', midpoint = 1.5)
print(minP, maxP)
predicted_y_points, predicted_y_std_points = predict_and_plot_points(
    uniformity_model, new_points_np, data_type = 'uniformity')

print('Predicted uniformity values:\n', predicted_y_points, '\nAnd their st devs:\n',
      predicted_y_std_points)
'''
###############################################################################
# Do the same as above for DFT and uniformity, to yellowness of the sample (est.
# by a human).

#yellowness_model = GP_model(visual_files, materials = materials, target_variable = 'Yellowness', lengthscale = 0.1, variance = 0.1)
with open(yellowness_model_path,'rb') as f:
    yellowness_model = pickle.load(f)

plot_GP(yellowness_model, x_points = yellowness_model.X, y_points = yellowness_model.Y, 
             data_type = 'yellowness')

minP, maxP = plot_P(yellowness_model, beta = 0.5, data_type = 'yellowness', midpoint = 1.5)
print(minP, maxP)

predicted_y_points, predicted_y_std_points = predict_and_plot_points(
    yellowness_model, new_points_np, data_type = 'yellowness')

print('Predicted yellowness values:\n', predicted_y_points, '\nAnd their st devs:\n',
      predicted_y_std_points)





