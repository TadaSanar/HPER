from hper_bo import load_ground_truth, predict_points
import numpy as np
from plotting_v2 import plot_mean_and_data, plot_std_only
def create_ternary_grid():

    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(0.0,1.0, 0.005)
    xt, yt, zt = np.meshgrid(a,a,a, sparse=False)
    points = np.transpose([xt.ravel(), yt.ravel(), zt.ravel()])
    points = points[abs(np.sum(points, axis=1)-1)<0.005]
    
    return points

bo_ground_truth_model_path = './Source_data/C2a_GPR_model_with_unscaled_ydata-20190730172222'
points = create_ternary_grid()
stability_model = load_ground_truth(bo_ground_truth_model_path)

# Scaled data. "GPy way"
all_preds = stability_model.predict(points)
# Data in px*min.
all_preds = predict_points(stability_model, points, Y_data = stability_model.Y)
# Data in px*h
all_preds = (all_preds[0]/60, # Mean
             all_preds[1]/360) # Variance 

opt_pred = np.min(all_preds[0])
idx_opt_pred = np.argmin(all_preds[0])
x_opt_pred = points[idx_opt_pred]
print('Model optimum location: ', x_opt_pred)


# Let's test if model gradients work as expected.
#stability_model.model.predictive gradients(x_opt_pred)

# Is the noise lower limit too high now? Check what we have for the final model.

plot_mean_and_data(points, all_preds[0], # These are for the surface.
                   np.array([x_opt_pred]), np.array([[opt_pred]]), # This is the point(s) plotted separately in the plot, the arrays need to be 2-dimensional.
                   color_lims = [np.min(all_preds[0]), np.max(all_preds[0])],
                   cbar_label =
                   r'$I_{c}(\theta)$ (px$\cdot$h)',
                   saveas = 'Mean_prediciton_with_samples')

plot_std_only(points, np.sqrt(all_preds[1]), saveas = 'Std_prediction_without_samples')
