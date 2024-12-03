#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import datetime
import GPy
import GPyOpt_DF as GPyOpt
#from hper_bo_simplified import create_grid, create_ternary_grid, predict_points

#script_dir = os.path.dirname(__file__)
#results_dir = os.path.join(script_dir, 'Results/')
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)

def predict_points(gpmodel, x_points, Y_data=None):
    '''
    For a GPy GP regression model or GPyOpt GPModel.
    '''
    if type(gpmodel) is GPy.models.gp_regression.GPRegression:
        
        # Prediction output is mean, variance.
        posterior_mean, posterior_var = gpmodel.predict_noiseless(x_points)
        posterior_std = np.sqrt(posterior_var)
        
    elif type(gpmodel) is GPyOpt.models.gpmodel.GPModel:
        
        # Prediction output of the GPModel is mean, standard deviation. So let's
        # dig out the GPRegression model and predict with that.
        posterior_mean, posterior_var = gpmodel.model.predict_noiseless(x_points)
        posterior_std = np.sqrt(posterior_var)
        
    # If the model has been trained with already-scaled (zero mean, unit
    # variance) data, the provided train data 'Y_data' will be used for scaling
    # the predictions to the correct units.
    if Y_data is not None:
        posterior_mean_true_units = posterior_mean * \
            np.nanstd(Y_data) + np.nanmean(Y_data)
        posterior_std_true_units = posterior_std * np.nanstd(Y_data)

        posterior_mean = posterior_mean_true_units
        posterior_var = posterior_std_true_units**2
        
    #print('\nPredict points noiseless: ', posterior_mean, 
    #      np.sqrt(posterior_var), '\n')
    
    return posterior_mean, posterior_var



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



def triangleplot(surf_points, surf_data, norm, surf_axis_scale = 1, cmap = 'RdBu_r',
                 cbar_label = '', saveas = None, surf_levels = None,
                 scatter_points=None, scatter_color = None, cbar_spacing = None,
                 cbar_ticks = None, close_figs = False):


    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    
    
    b=surf_points[:,0]
    c=surf_points[:,1]
    a=surf_points[:,2]
    # values stored in the last column
    v = np.squeeze(surf_data)/surf_axis_scale#[:,-1]/surf_axis_scale
    # translate the data to cartesian
    x = 0.5 * ( 2.*b+c ) / ( a+b+c )
    y = 0.5*np.sqrt(3) * c / (a+b+c)
    # create a triangulation
    T = tri.Triangulation(x,y)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(3.6/2.54)
    fig.set_figwidth(5/2.54)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.14, top=0.91)

    if surf_levels is None:
        
        # Default value
        nlevels = 26
        
        minvalue = norm.vmin-(np.abs(norm.vmin)/1000)
        maxvalue = norm.vmax + (np.abs(norm.vmax)/1000)
        
        # If 0 is less than 2 steps away from the lower end of the colorbar 
        # range, default the colorbar lower end to 0 (because it looks more tidy).
        step = (maxvalue - minvalue) / nlevels
        if (minvalue > 0) and (minvalue < (2 * step)):
        
            minvalue = 0
            step = (maxvalue - minvalue) / nlevels
        
        surf_levels = np.arange(minvalue, maxvalue + step, step)
        
        # Every 5th surf level will have a tick mark so they need to be fixed
        # to a two digit value while ensuring the contour levels increase.
        
        tick_idx = np.arange(1,nlevels+1,5)
        
        if (norm.vmax > 0):
            
            if (np.log10(step) > -2):
                n_sign_digits = 2 # Works for almost all the values because there is exp scale in the colorbar.
            elif step != 0: # Very small values, need more accuracy in surf level stepping.
                n_sign_digits = np.abs(np.floor(np.log10(step)))
            else:
                n_sign_digits = 2
            
            if (np.floor(np.log10(norm.vmax)) - np.floor(np.log10(norm.vmin)) 
                != 0):
                
                # Standard case, "the range of values is large"
                decimals_to_round = -int(np.floor(np.log10(norm.vmax)) - 
                                         n_sign_digits)
                
            else:
                
                # "The range of values is small"
                decimals_to_round = -int(np.floor(np.log10(norm.vmax)) -
                                         n_sign_digits - 1)
                
            surf_levels[tick_idx[0:-1]] = np.round(surf_levels[tick_idx[0:-1]], 
                                                  decimals = decimals_to_round)
            
            # Make sure the last tick index is close to the true upper end of 
            # the range (the above rounding might go too far above the upper 
            # end of the range).
            decimals_to_round = -int(np.ceil(np.log10(norm.vmax)) - 
                                     n_sign_digits - 1)
            surf_levels[tick_idx[-1]] = np.round(surf_levels[tick_idx[-1]], 
                                                  decimals = decimals_to_round)
            
        elif norm.vmax != 0:
            
            surf_levels[tick_idx[0:-1]] = np.round(surf_levels[tick_idx[0:-1]], 1)
            
        if any(surf_levels[1::]-surf_levels[0:-1] <= 0):
            
            print('Problem with plotting surface levels: ', surf_levels, '\n')
            surf_levels = np.arange(minvalue, maxvalue + step, step)
            #raise Exception('Problem with plotting surface levels. Set them manually.')
    
    else:
        
        # Surf levels have been given, just specify the tick idx.
        tick_idx = np.arange(len(surf_levels))
    
    if ((np.isnan(v).any()) | (np.isnan(x).any()) | (np.isnan(y).any())) == True:
        
        print(v)
        print(x)
        print(y)
    
    # plot the contour
    im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap, levels=surf_levels, norm = norm)
    
    if (scatter_points is not None):
        
        #Triangulation for the points suggested by BO.
        b_p = scatter_points[:,0]
        c_p = scatter_points[:,1]
        a_p = scatter_points[:,2]
        x_p = 0.5 * ( 2.*b_p+c_p ) / ( a_p+b_p+c_p)
        y_p = 0.5*np.sqrt(3) * c_p / (a_p+b_p+c_p)
        
        im3 = ax.scatter(x_p, y_p, s=8, c=scatter_color, cmap=cmap,
                         edgecolors='black', linewidths=.5, alpha=1, zorder=2,
                         norm=norm)
    
    myformatter=mpl.ticker.ScalarFormatter()
    myformatter.set_powerlimits((0,2))
    if (cbar_spacing is not None) and (cbar_ticks is not None):
        cbar=plt.colorbar(im, ax=ax, spacing=cbar_spacing, ticks=cbar_ticks)
    elif cbar_ticks is not None:
        cbar=plt.colorbar(im, ax=ax, ticks=cbar_ticks)
    else:
        cbar=plt.colorbar(im, ax=ax, format=myformatter, #spacing=surf_levels, 
                          ticks=surf_levels[tick_idx])
        
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=8)#, labelpad = -0.5)
    plt.axis('off')
# =============================================================================
#     Values used in C2a:
#     plt.text(0.35,-0.1,'Cs (%)')
#     plt.text(0.10,0.54,'FA (%)', rotation=61)
#     plt.text(0.71,0.51,'MA (%)', rotation=-61)
#     plt.text(-0.0, -0.1, '0')
#     plt.text(0.87, -0.1, '100')
#     plt.text(-0.07, 0.13, '100', rotation=61)
#     plt.text(0.39, 0.83, '0', rotation=61)
#     plt.text(0.96, 0.05, '0', rotation=-61)
#     plt.text(0.52, 0.82, '100', rotation=-61)
#     
# =============================================================================
    plt.text(0.35,-0.1,'Cs (%)')
    plt.text(0.09,0.34,'FA (%)', rotation=61)
    plt.text(0.71,0.31,'MA (%)', rotation=-61)
    plt.text(-0.0, -0.1, '0')
    plt.text(0.87, -0.1, '100')
    plt.text(-0.08, 0.05, '100', rotation=61)
    plt.text(0.38, 0.83, '0', rotation=61)
    plt.text(0.96, 0.05, '0', rotation=-61)
    plt.text(0.52, 0.75, '100', rotation=-61)
    
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=0)
    #plotting the mesh
    im2=ax.triplot(trimesh,'k-', linewidth=0.5)
    
    plt.tight_layout()
    
    if saveas:
        #fig.savefig(saveas + '.pdf', transparent = True)
        #fig.savefig(saveas + '.svg', transparent = True)
        fig.savefig(saveas + '.png', dpi=300)
    
    if close_figs:
        
        plt.close()
        
    else:
        
        plt.show()
    
    return fig, ax

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    ## TO DO: This is a modified version of an open access function. Deal with ack.
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), #128, endpoint=False), 
        np.linspace(midpoint, 0.3, 80, endpoint=False),
        np.linspace(0.3, 1.0, 49, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    if name not in list(mpl.colormaps):
        newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
        mpl.colormaps.register(cmap=newcmap)
    else:
        
        raise Exception('A new colormap with the same name cannot be created.')

    return newcmap
'''
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
'''
def init_plots(rounds, limit_file_number, time_str, results_folder):
    
    # This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_ternary_grid()
    
    mean = [None for k in range(rounds)]
    std = [None for k in range(rounds)]
    acq = [None for k in range(rounds)]
    ref_acq = [None for k in range(rounds)]
    
    #original_folder = os.getcwd()
    #os.chdir(original_folder)
    if time_str is None:
        time_now = '{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
    else:
        time_now = time_str
        
    results_dir = results_folder + 'Triangles/'+ time_now + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if (limit_file_number == True) and (rounds > 10):
        
        # Limit the number of plotting files to 5/BO cycle/plot type.
        # Additionally, only the most essential plots will be created.
        
        rounds_to_plot = [0, 2, 10, int(np.floor(rounds/2)), (rounds-1)]
    
    else:
        
        rounds_to_plot = range(rounds)
    
    return points, mean, std, acq, ref_acq, time_now, results_dir, rounds_to_plot

def fill_ternary_grid(mean, std, GP_model, points, y_train_data = None):
    
    # This function can deal with GPy GPRegression model and GPyOpt
    # GPModel, not other models.
    posterior_mean, posterior_var = predict_points(GP_model, points, 
                                                   Y_data = y_train_data)
    
    mean = posterior_mean
    std = np.sqrt(posterior_var)
    
    return mean, std
    
'''
def fill_ternary_grid(mean, std, GP_model, points, y_train_data = None):
    
    # : Here the posterior mean and std_dv+acquisition function are calculated.
    mean, posterior_var = GP_model.predict(points) # MAKING THE PREDICTION, GPy GPregression model assumed
    
    if y_train_data is not None:
        
        # Scaling the normalized data back to the original units (GPyOpt BO scales the data before training the model).
        mean = mean*y_train_data.std()+y_train_data.mean()
        std = np.sqrt(posterior_var)*y_train_data.std()
    
    else:
        
        std = np.sqrt(posterior_var)
        
    return mean, std
'''

def fill_ternary_grids(mean, std, acq, ref_acq, rounds, BO_batch, points, 
                       y_train_data = None, scale_acq = True):
    
    if ref_acq != None:
        return_ref_acq = True
    
    for i in range(rounds):
        
        if y_train_data is None:
            y_t = None
        else:
            y_t = y_train_data[i]
        
        # : Here the posterior mean and std_dv+acquisition function are calculated.
        mean[i], std[i] = fill_ternary_grid(mean[i], std[i], 
                                            BO_batch[i].model.model, points, 
                                            y_train_data = y_t)
        
        acq_i=BO_batch[i].acquisition.acquisition_function(points)
        
        if scale_acq is True:
            
            # Scaling the acquisition function to btw 0 and 1.
            acq[i] = (-acq_i - min(-acq_i))/(max(-acq_i - min(-acq_i)))
            
        else:
            
            acq[i] = acq_i
            
        if return_ref_acq == True:
            
            from GPyOpt.acquisitions.LCB_DF import AcquisitionLCB_DF
            from GPyOpt.acquisitions.EI_DF import AcquisitionEI_DF
            from GPyOpt.acquisitions.EI_noisy_DF import AcquisitionEI_noisy_DF
            
            if type(BO_batch[i].acquisition) is AcquisitionLCB_DF:
                
                from GPyOpt.acquisitions.LCB import AcquisitionLCB
                model = BO_batch[i].acquisition.model
                expl_w = BO_batch[i].acquisition.exploration_weight
                space = BO_batch[i].acquisition.space
                optimizer = BO_batch[i].acquisition.optimizer
                ref_acq_obj = AcquisitionLCB(model=model, space=space, optimizer=optimizer, 
                                  exploration_weight=expl_w)
                ref_acq_i=ref_acq_obj.acquisition_function(points)
                
            elif type(BO_batch[i].acquisition) is AcquisitionEI_DF:
                
                from GPyOpt.acquisitions.EI import AcquisitionEI
                model = BO_batch[i].acquisition.model
                jitter = BO_batch[i].acquisition.jitter
                space = BO_batch[i].acquisition.space
                optimizer = BO_batch[i].acquisition.optimizer
                ref_acq_obj = AcquisitionEI(model=model, space=space, optimizer=optimizer, 
                                  jitter=jitter)
                
                ref_acq_i=ref_acq_obj.acquisition_function(points)
                
            elif type(BO_batch[i].acquisition) is AcquisitionEI_noisy_DF:
                
                from GPyOpt.acquisitions.EI_noisy import AcquisitionEI_noisy
                model = BO_batch[i].acquisition.model
                jitter = BO_batch[i].acquisition.jitter
                space = BO_batch[i].acquisition.space
                optimizer = BO_batch[i].acquisition.optimizer
                ref_acq_obj = AcquisitionEI_noisy(model=model, space=space, optimizer=optimizer, 
                                  jitter=jitter)
                
                ref_acq_i=ref_acq_obj.acquisition_function(points)
                
            else:
                
                ref_acq_i = None
                
            if (scale_acq is True) and (ref_acq_i is not None):
                    
                # Scaling the acquisition function to btw 0 and 1.
                ref_acq[i] = (-ref_acq_i - min(-ref_acq_i))/(max(-ref_acq_i - min(-ref_acq_i)))
                    
            else:
                    
                ref_acq[i] = ref_acq_i
                
    return mean, std, acq, ref_acq
    
    
def save_round_to_csv_files(mean, std, acq, rounds, materials, points,
                            limit_file_number, results_dir, time_now,
                            x_data, y_data, next_suggestions = None,
                            mean_name = 'Ic (px*min)',
                            std_name = 'Std of Ic (px*min)', acq_name = 'EIC'):

    for i in range(rounds):

        if (limit_file_number == False) and (next_suggestions is not None):        
            next_suggestions[i].to_csv(results_dir + 
                                       'Bayesian_suggestion_round_'+str(i) + 
                                       time_now + '.csv', float_format='%.3f')
        
        # Create DataFrame. Works also for empty DataFrames.
        inputs = pd.DataFrame(columns = x_data[i].columns.append(
            pd.Index([mean_name])))
        if x_data[i].empty is False:
            inputs = pd.concat((inputs, x_data[i])) 
            inputs.loc[:, mean_name] = y_data[i].values
        inputs=inputs.sort_values(mean_name)
        #inputs=inputs.drop(columns=['Unnamed: 0'], errors='ignore')

        if (limit_file_number == False):
            inputs.to_csv(results_dir + 'Model_inputs_round_'+str(i)+ 
                          time_now + '.csv', float_format='%.3f')
        
        inputs2 = pd.DataFrame(points, columns=materials)
        inputs2[mean_name]=mean[i]
        inputs2[std_name]=std[i]
        inputs2[acq_name]=acq[i]

        if (limit_file_number == False):
            inputs2.to_csv(results_dir + 'Model_round_' + str(i) + 
                           time_now + '.csv', float_format='%.3f')

def define_norm_for_surf_plot(target, color_lims = None):
    
    if color_lims is not None:
        
        lims = color_lims
        
    else:
        lims = [np.nanmin(target), np.nanmax(target)]
    
    if lims[0] == np.nan: # There is no data
        lims[0] = 0
        
    if lims[1] == np.nan: # There is no data
        lims[1] = 0
        
    if (lims[0] == 0) and (lims[1] == 0): # There's only zero data.
            
        lims[0] = -0.5
        lims[1] = 0.5
        
    if lims[0] == lims[1]: # There's only constant data (that is nonzero).
            
        lims[0] = lims[0] - np.abs(lims[0]/2)
        lims[1] = lims[1] + np.abs(lims[1]/2)
    
    norm = mpl.colors.Normalize(vmin=lims[0], vmax=lims[1])
    
    return norm
    
def plot_surf_with_lims_and_name(points, target, color_lims, cmap, cbar_label,
                                     saveas, cbar_ticks = None, close_figs = False):
    
    norm = define_norm_for_surf_plot(target, color_lims = color_lims)
    
    triangleplot(points, target, norm, cmap = cmap,
                 cbar_label = cbar_label, saveas = saveas,  
                 cbar_ticks = cbar_ticks, close_figs = close_figs)

    
def plot_mean_only(points, mean, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', saveas = None,
                   close_figs = False):
                
    plot_surf_with_lims_and_name(points, mean,
                                     color_lims = color_lims, cmap = cmap,
                                     cbar_label = cbar_label, saveas = saveas,
                                     close_figs = close_figs)

def plot_std_only(points, std, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'Std $I_{c}(\theta)$ (px$\cdot$h)', 
                   saveas = None,  cbar_ticks = None, close_figs = False):
                
    plot_surf_with_lims_and_name(points, std,
                                     color_lims = color_lims, cmap = cmap,
                                     cbar_label = cbar_label, saveas = saveas,  
                                     cbar_ticks = cbar_ticks, close_figs = close_figs)

def init_acq_plot(points, acq, cmap_base):

    if cmap_base == 'RdBu':
    	
        orig_cmap = mpl.cm.RdBu
    	# Shift the colormap (in order to remove the red background that appears with the standard version of the colormap).
        cmap_name = 'shifted_cm'
        
        if not(cmap_name in list(mpl.colormaps)):
            
            shiftedColorMap(orig_cmap, start=0, midpoint=0.000005, stop=1, name=cmap_name)
            
    else:
    	
    	# Use user defined colormap and do nothing else.
    	cmap_name = cmap_base
    	
    if acq is None:
    	
    	# Plot the starting point of the acquisitions i.e. an even distribution.
        acq_to_plot = np.ones((points.shape[0], 1))*0.3
    
    else:
        acq_to_plot = acq
        
        ## Old version:
        #if i == 0:
        #    # In the first round there is an even distribution.
        #    test_data = np.concatenate((points, acq_normalized[i]), axis=1)
        #    test_data = test_data[test_data[:,3] != 0]
        #    test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
        #else:
        #    test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
        #    test_data = test_data[test_data[:,3] != 0]
    	
    # Acquisition function may have a lot of zero values. The plot looks
    # pretty weird unless they are removed.
    all_data = np.concatenate((points, acq_to_plot), axis = 1)
    all_data = all_data[all_data[:,3] != 0]
    acq_to_plot = all_data[:,3]
    points_to_plot = all_data[:,0:3] 
    
    plot_params = {
        'surf_axis_scale': 1.0,
        'cmap_name': cmap_name,
        'surf_levels': [0,0.009,0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
        'cbar_spacing': 'proportional',
        'cbar_ticks': [0,0.2,0.4,0.6,0.8,1]
        }
    
    return points_to_plot, acq_to_plot, plot_params
    
def plot_acq_only(points, acq, color_lims = None, cmap_base = 'RdBu',
                   cbar_label = r'$Acq(\theta)$', saveas = None, close_figs = False):
    
    # Acq. colorbar label in C2a:
    # r'$EIC (\theta, \beta_{DFT})$'
    
    # Acq.fun. plot looks weird unless zeros are removed and colormap shifted. 
    points_to_plot, acq_to_plot, plot_params = init_acq_plot(points, acq, cmap_base)
    
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Repeat its functionality, instead.
    
    # Note that color lims does not include zeros (since they are removed at
    # two lines above this point) unless the user feeds in color_lims with zero
    # included.
    norm = define_norm_for_surf_plot(acq_to_plot, color_lims = color_lims)
    
    triangleplot(points_to_plot, acq_to_plot, norm,
                 surf_axis_scale = plot_params['surf_axis_scale'],
                 cmap = plot_params['cmap_name'],
                 cbar_label = cbar_label,
                 saveas = saveas, 
                 surf_levels = plot_params['surf_levels'],
                 cbar_spacing = plot_params['cbar_spacing'],
                 cbar_ticks = plot_params['cbar_ticks'], close_figs = close_figs)
    
def plot_acq_and_data(points, acq, data, color_lims = None, cmap_base = 'RdBu',
                   cbar_label = r'$Acq(\theta)$', saveas = None, close_figs = False):
    
    
    # Acq.fun. plot looks weird unless zeros are removed and colormap shifted. 
    points_to_plot, acq_to_plot, plot_params = init_acq_plot(points, acq, cmap_base)
    
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Code again, instead.
    
    # Note that color lims does not include zeros (since they are removed at
    # two lines above this point) unless the user feeds in color_lims with zero
    # included.
    norm = define_norm_for_surf_plot(acq_to_plot, color_lims = color_lims)
    
    # Colors of the samples.     
    # Old version in C2a:
    #newPal = {} #0:'k', 1:'k', 2:'k', 3:'k', 4: 'k'}#{0:'#8b0000', 1:'#7fcdbb', 2:'#9acd32', 3:'#ff4500', 4: 'k'} #A14
    #for i in rounds_to_plot:
    #    newPal[i] = 'k'
    
    triangleplot(points_to_plot, acq_to_plot, norm,
                 surf_axis_scale = plot_params['surf_axis_scale'],
                 cmap = plot_params['cmap_name'],
                 cbar_label = cbar_label,
                 saveas = saveas, 
                 surf_levels = plot_params['surf_levels'],
                 scatter_points=data, scatter_color = 'k',
                 cbar_spacing = plot_params['cbar_spacing'],
                 cbar_ticks = plot_params['cbar_ticks'], close_figs = close_figs)
    
def plot_mean_and_data(points, mean, data_x, data_y, color_lims = None, cmap = 'RdBu_r',
                   cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', saveas = None,
                   cbar_ticks = None, close_figs = False):
                
    # Let's not use the convenience function 'plot_surf_with_lims_and_name'
    # because we change so many parts. Repeat its functionality, instead.
    
    # Norm calculation needs to take account the mean data and sampled data.
    norm = define_norm_for_surf_plot(np.append(mean, data_y), 
                                     color_lims = color_lims)
    triangleplot(points, mean, norm, 
                 cmap = cmap,
                 cbar_label = cbar_label, 
                 saveas = saveas,
                 scatter_points = data_x, 
                 scatter_color = np.ravel(data_y),
                 #cbar_spacing = None, cbar_ticks = cbar_ticks)
                 cbar_ticks = cbar_ticks, close_figs = close_figs)

def plotBO(rounds, suggestion_df, #compositions_input, degradation_input,
           BO_objects, materials, X_rounds, Y_rounds, Y_accum, X_accum, x_next,
           limit_file_number = True, time_str = None, 
           results_folder = './Results/', minimize = True, close_figs = False):
    
    # Create a ternary grid 'points', the necessary folder structure, and 
    # file name templates. Initialize the posterior mean and st.dev., and
    # acquisition function lists.
    points, posterior_mean, posterior_std, acq_normalized, ref_acq_normalized, time_now, results_dir, rounds_to_plot = init_plots(
        rounds, limit_file_number, time_str, results_folder)
    
    # Fill in the lists with surfaces to plot.
    posterior_mean, posterior_std, acq_normalized, ref_acq_normalized = fill_ternary_grids(posterior_mean, 
                                                                       posterior_std, 
                                                                       acq_normalized, 
                                                                       ref_acq_normalized, 
                                                                       rounds, 
                                                                       BO_objects, 
                                                                       points, 
                                                                       y_train_data = Y_accum)
    
    
    ###############################################################################
    # SAVE RESULTS TO CSV FILES
    ###############################################################################
    
    # Save csvs only if limit_file_number is False.
    save_round_to_csv_files(posterior_mean, posterior_std, acq_normalized,
                            rounds, materials, points,
                                limit_file_number, results_dir, time_now,
                                x_data = X_rounds, #compositions_input,
                                y_data = Y_rounds, #degradation_input,
                                next_suggestions = suggestion_df)
            
    ###############################################################################
    # PLOT
    ###############################################################################
    # Let's plot the resulting suggestions. This plot works for 3 materials only.
    
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    
    axis_scale = 60 # Units from px*min to px*hour.
    
    # Posterior limits need to take account both mean data and the real samples.
    mins = [Y_accum[i].min() for i in list(range(len(Y_accum)))]
    maxs = [Y_accum[i].max() for i in list(range(len(Y_accum)))]
    lims_samples = [np.min(mins)/axis_scale, np.max(maxs)/axis_scale]
    lims_posterior = [np.min(posterior_mean)/axis_scale, np.max(posterior_mean)/axis_scale]
    
    lims_p = [np.min([lims_samples[0], lims_posterior[0]]), 
              np.max([lims_samples[1], lims_posterior[1]])]
    lims_s = [np.min(posterior_std)/axis_scale, 
              np.max(posterior_std)/axis_scale]
    lims_a = [0,1]
    
    for i in rounds_to_plot:
        
        # Plot acquisition function at the beginning of the round with samples
        # suggested for the round.
        
        if i == 0:
            acq_to_plot = None # At the beginning, we have an even distribution.
            ref_acq_to_plot = None
        else:
            acq_to_plot = acq_normalized[i-1]
            ref_acq_to_plot = ref_acq_normalized[i-1]
            # Note i-1, this is because acq is saved at the end of each round
            # and here we plot acq at the beginning of each round.
        
        plot_acq_and_data(points, acq_to_plot, X_rounds[i].values, color_lims = lims_a,
                          cbar_label = r'$Acq(\theta)$ in round ' + str(i),
                          saveas = results_dir +
                          'Acq-with-single-round-samples-round'+str(i) + 
                          '-' + time_now, close_figs = close_figs)
        
        # Plot posterior mean with samples acquired by the end of the round.
        plot_mean_and_data(points, posterior_mean[i]/axis_scale,
                           X_accum[i], Y_accum[i]/axis_scale,
                           color_lims = lims_p,
                           cbar_label =
                           r'$I_{c}(\theta)$ (px$\cdot$h) in round ' + str(i),
                           saveas = results_dir +
                           'Modelled-Ic-with-samples-round' + str(i) + 
                           '-' + time_now, close_figs = close_figs)
        
        plot_std_only(points, posterior_std[i]/axis_scale,
                      color_lims = lims_s,
                      saveas = results_dir + 'St-Dev-of-modelled-Ic-round' +
                      str(i) + '-' + time_now, close_figs = close_figs)
        
        if ref_acq_to_plot is not None:
            plot_acq_and_data(points, ref_acq_to_plot, X_rounds[i].values, 
                              color_lims = lims_a, 
                              cbar_label = r'$Ref acq(\theta)$ in round ' + str(i),
                              saveas = results_dir +
                              'Ref-acq-with-single-round-samples-round'+str(i) + 
                              '-' + time_now, close_figs = close_figs)
        
        if (limit_file_number == False):        
            
            plot_mean_only(points, posterior_mean[i]/axis_scale,
                           color_lims = lims_p,
                           saveas = results_dir + 'Modelled-Ic-no-grid-round' +
                           str(i) + '-' + time_now, close_figs = close_figs)
            
            plot_acq_only(points, acq_to_plot,
                                      color_lims = lims_a,
                                      cbar_label = r'$Acq(\theta)$ in round ' + str(i),
                                      saveas = results_dir + 'Acq-round'+str(i) + 
                                     '-' + time_now, close_figs = close_figs)
            
    plt.close('all')
    
