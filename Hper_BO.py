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

    # Mean predictions.
    posterior_mean, posterior_std = predict_points(GP_model, x_points, Y_train = Y_train)
    
    # Adding noise to the predictions.
    noise = np.random.rand(posterior_mean.shape[0], posterior_mean.shape[1]) # Values between 0 and 1.
    noise_scaled = (noise - 0.5)*2.0
    
    posterior_mean_noisy = posterior_mean + noise_scaled*posterior_std
    
    return posterior_mean_noisy, posterior_mean, posterior_std    

def predict_points_noisy_c2a(GP_model, x_points, y_ground_truth):
    
    mean_noisy, mean, std = predict_points_noisy(GP_model, x_points, Y_train = y_ground_truth)
    
    return mean_noisy



#%%
###############################################################################

# These variables are related to data collection and computing the figure of merit.
# Scroll down for changing variables related to the Bayesian Optimization.
        
# Give path to the folders that contain camera CSV data. The data will be
# analyzed in this order. Include '/' or '\' from the end of the string.
original_folder = os.getcwd()

folders = [r'/20190606-R1-JT/BMP/RGB/Calibrated/',
           r'/20190622-R1-JT/BMP/RGB/Calibrated/',
           r'/20190711-R1-JT/BMP/RGB/Calibrated/',
           r'/20190723-R1-JT/BMP/RGB/Calibrated/',
           r'/20190809-R1-JT/BMP/RGB/Calibrated/']

for n in range(len(folders)):
    folders[n] = original_folder + folders[n]     

########
# Clean up later. C2a end model.

stability_exp_results = 'Backup-model-20190730172222' # This pickle file contains all the results from the C2a optimization.

# Load already existing stability data as the "ground truth" of stability.
with open(stability_exp_results,'rb') as f:
    [BO_batch_objects, next_sugg_df, x_next_sugg, X_rounds, Y_rounds] = pickle.load(f)
stability_model = BO_batch_objects[-1].model
x_stability = np.concatenate(X_rounds)
y_stability = np.concatenate(Y_rounds)

# Sample from the ground truth function instead of the folders given above.
# If "function = True", the search is started from fixed points (this is the
# current implementation, to do: grid points?).
function = True

# Give the number of BO rounds repeated. Will be repeated as many times as the
# number of data files that the user feeds in if "rounds = None" (this setting
# is meant for actual experimental optimization rounds, i.e. when "function = False").
rounds = 5

if rounds == None:
    rounds = len(folders)

# Give the number of suggestions given for the next sampling points. Will be set
# to the number of samples in the data files if "n_batch = None" (this setting
# is meant for actual experimental optimization rounds, i.e. when "function = False",
# will throw an error if "function = True").
n_batch  = 1

########


# Give True if the data is calibrated, and False if the data is raw.
is_calibrated = True
# Give True if the data is RGB, and False if the data is LAB.
is_rgb = True



# Give the materials the compositions of which are being optimized. Use the
# same format than in the 'Samples.csv' of the aging tests.
materials = ['CsPbI', 'MAPbI', 'FAPbI']
# Note: Current implementation of tolerance factor function works only for
# these materials. Current implementation of solubility/DFT works only for
# these materials in this order.

# Give the cutoff (in minutes) for analyzing the data. The firs n minutes will
# be utilized and the rest of the data is dropped off.
cutoff = 7000

# We need to drop the empty sample slots from the analysis. This is done by
# by searching for an indicator string from 'Samples.csv', column 'Sample'. The
# default value is '---'.
indicator_for_empty = '---' # Or 'NA'

# Choose how the possible duplicate samples (same compositions several times)
# are treated. Write one of these options: 'first' (choose the first one of the
# duplicates), 'last' (choose the last one of the duplicates), 'mean'
# (calculate the mean of the duplicates, remove the original samples, and
# replace by the mean value), or 'full' (pass all the data to the BO function
# as such - the BO function takes into account every sample). This will treat
# only duplicates in each round, not between the rounds. Default value is 'full'.
duplicate_operation = 'full'

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

batch_size = [None for j in range(rounds)]

x_next = [None for j in range(rounds)] # Suggestions for the next locations to be sampled.
suggestion_df = [None for j in range(rounds)] # Same in nice dataframes.

BO_batch = [None for j in range(rounds)] # Batch BO objects for each BO round (with data acquired by that time).

current_data_fusion_files = [] # Will be filled in later on.

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

if function is True: # C2a function! TO do: clean up
    
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
        
        # TO DO: Generalize to more than three materials or shift up to the user defined part.
        constraints[k] = [{'name': 'constr_1', 'constraint': 'x[:,0] +x[:,1] + x[:,2] - ' + str(composition_total[1])},
                       {'name': 'constr_2', 'constraint': str(composition_total[0])+'-x[:,0] -x[:,1] - x[:,2] '},
                       {'name': 'constr_3', 'constraint': tolerance_factor_function}]
        
        batch_size[k] = n_batch
        
    for j in range(len(materials)):
        bounds[j] = {'name': materials[j], 'type': 'continuous', 'domain': (0,1)}





os.chdir(original_folder)

###############################################################################
# These variables are related to the Bayesian optimization.
num_cores = 1 # Not a parallel run
jitter = 0.01 # The level of exploration.

# Data fusion constraint definition.

# These files contain DFT data that is integrated into the optimization loop as
# a soft constraint (starting from the round it is first listed - if the vector
# is shorter than the number of rounds, it is assumed that no data is being
# added to data fusion in the rounds not listed).
dft_files = [['/phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
             '/phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
             '/phasestability/CsMA/fulldata/CsMA_T300_above.csv']
             ]
dft_variable = 'dGmix (ev/f.u.)'
# These depend on the data fusion target variable you choose.
dft_lengthscale = 0.03 # For Gaussian process regression
dft_variance = 2 # For GPR
dft_beta = 0.025 # For probability
dft_midpoint = 0 # For P

# Visual quality of the samples as a constraint.
visual_files = [['/visualquality/visualquality_round_0.csv'],
                 ['/visualquality/visualquality_round_1.csv'],
                 ]
visual_variable = 'Yellowness'
visual_lengthscale = 0.1
visual_variance = 0.1
visual_beta = 0.5
visual_midpoint = 1.5

# Arbitrary function (such as a direct cutoff) as a constraint.
cutoff_files = [['/arbitrary_function/cutoff.csv']]
cutoff_variable = 'Cutoff'
# No GPR fit or P scaling is done here for the function in this implementation.

# Please choose one of the above as the constraint.
data_fusion_files = dft_files
data_fusion_variable = dft_variable
lengthscale = dft_lengthscale
variance = dft_variance
beta = dft_beta
midpoint = dft_midpoint

###############################################################################
#%%
# BEGIN BAYESIAN OPTIMIZATION


for k in range(rounds):
        
    if len(data_fusion_files) > k:
        
        for n in range(len(data_fusion_files[k])):
            current_data_fusion_files.append(original_folder + data_fusion_files[k][n])

    
    if (k == 0) and (function == True):
        # Random initialization 2D
        #compositions_input = [pd.DataFrame(np.random.rand(batch_size[k], len(materials))*20-10, columns = materials)]
        #rand_init = np.random.rand(batch_size[k], len(materials))*20-10 # 2D, [10,10]
        grid_init = np.array([[0.33, 0.33, 0.33], [0,0,1]])#1, 0, 0]])#, [0,1,0], [0,0,1], [0.5,0.5,0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) # To do: generalize
        #rand_init = np.random.rand(batch_size[k], len(materials)) # Ternary
        #for i in range(len(materials)):
        #    if i == 0:
        #        rand_init = np.random.rand(batch_size[k], len(materials))
        #    if i == 1:
        #        for l in range(batch_size[k]):
        #            if rand_init[l,i] + rand_init[l,0] > 1:
        #                rand_init[l,i] = 1 - rand_init[l,0]
        #    if i == 2:
        #        rand_init[:,i] = 1 - rand_init[:,0] - rand_init[:,1]        
        compositions_input = [pd.DataFrame(grid_init, columns = materials)]
        #constraints[k] = None
        #batch_size[k] = 3
        degradation_input = []
        
    if (k > 0) and (function == True):
        # The locations suggested after the previous BO round will be sampled in this round.
        #compositions_input.append(compositions_input[k-1].append(pd.DataFrame(x_next[k-1], columns = materials), ignore_index=True))
        compositions_input.append(pd.DataFrame(x_next[k-1], columns = materials))
        #constraints[k] = None
        
    df = compositions_input[k].copy()
    
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
            
    # Function merit. TO DO: clean up so that works for any #D
    if function == True:
        x = df.iloc[:,0:len(materials)].values
        ##x2 = df.iloc[:,1].values
        #df['Merit'] = f_booth(x)
        df['Merit'] = predict_points_noisy_c2a(stability_model, x, y_stability)
        
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
    print('X_step and Y_step for round ' + str(k) + ':', X_step[k], Y_step[k])
    #Define Bayesian Opt object
    #Instantiate BO object, f=None as we are using tabular data, no analytical function
    BO_batch[k] = GPyOpt.methods.BayesianOptimization(f=None,#f_booth,  
                                            domain = bounds,
                                            constraints = constraints[k],
                                            acquisition_type = 'EI_DFT',
                                            files = current_data_fusion_files,
                                            data_fusion_target_variable = data_fusion_variable,
                                            normalize_Y = True,
                                            X = X_step[k],
                                            Y = Y_step[k],
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size[k],
                                            acquisition_jitter = jitter,
                                            lengthscale = lengthscale,
                                            variance = variance,
                                            beta = beta,
                                            midpoint = midpoint,
                                            data_fusion_input_variables = materials)    
    #Suggest next points (samples to prepare).
    x_next[k] = BO_batch[k].suggest_next_locations()
    suggestion_df[k] = pd.DataFrame(x_next[k], columns = materials)
    suggestion_df[k]['Total'] = suggestion_df[k].sum(axis = 1)
    #suggestion_df[k]['Tolerance Factor'] = tolerance_factor(
    #        suggestion_df = suggestion_df[k],
    #        tolerance_factor_bound = None)
    BO_batch[k].plot_acquisition() # Works only for max 2D.

# Plot and save the results.

#if function == True:
#    degradation_input = compositions_input

plotBO(rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step)    

print('Results are saved into folder ./Results.')

# Save the model as an backup
# dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab') 
# pickle.dump([BO_batch, suggestion_df, x_next, X_rounds, Y_rounds], dbfile)                      
# dbfile.close()

# Minimum value vs rounds.
optimum = []
for i in range(rounds):
    optimum.append(np.min(Y_step[i]))

plt.figure()
plt.plot(range(rounds), optimum)
plt.show()
