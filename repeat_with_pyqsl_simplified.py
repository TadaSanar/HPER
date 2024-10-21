import psutil
import pickle
from set_figure_defaults import FigureDefaults
from hper_bo import bo_sim_target
from hper_util_repetitions_lcb import cg, build_filenames, set_bo_settings, set_repeat_settings_simplified, modify_filename
from hper_util_gp import load_GP_model
import pyqsl
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
#
fig_def = FigureDefaults(style='presentation')

# Task function


def bo_args_to_dict(n_repetitions, n_rounds, n_init, batch_size, materials, noise_target, acquisition_function):
    return {
        "n_repetitions": n_repetitions,
        "n_rounds": n_rounds,
        "n_init": n_init,
        "batch_size": batch_size,
        "materials": materials,
        "noise_target": noise_target,
        "acquisition_function": acquisition_function
    }


def repeats_to_range(n_repetitions):
    return list(range(n_repetitions))


def task(m, c_g, c_exclz, c_eig, jitter, bo_params, noise_df, folder,
         init_points, indices_of_repeats, save_figs, gt_model_targetprop,
         gt_model_human, close_figs, save_disk_space, ground_truth_rA,
         debug_mode_printouts=True, additional_idx_for_folder=None
         ):

    data_fusion_property, df_data_coll_method, acquisition_function, c_grad, c_e = set_repeat_settings_simplified(
        m, c_g, c_exclz, c_eig, bo_params)
    outputs = {}

    ###############
    # Typically, one does not need to modify these inputs.
    acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params = set_bo_settings(
        bo_params, acquisition_function, jitter, data_fusion_property,
        df_data_coll_method, noise_df, c_grad, c_e)

    # Create result folders and build filenames for result files.
    pickle_filenames, figure_filenames, triangle_folder = build_filenames(
        folder, bo_params, acq_fun_descr, df_data_coll_descr,
        fetch_file_date=None, m=m, additional_idx=additional_idx_for_folder)

    ###############

    all_starting_points = []
    bo_examples = []
    optima = []
    model_optima = []
    X_accum_all = []
    Y_accum_all = []
    data_fusion_params_all = []
    surrogate_model_params_all = []
    base_model_values = []
    regrets = []
    regrets_samples = []

    lengthscales_target_all = []
    variances_target_all = []
    gaussian_noise_variances_target_all = []
    X_accum_df_final = []
    Y_accum_df_final = []
    df_optima = []
    df_model_optima = []
    lengthscales_df_all = []
    variances_df_all = []
    gaussian_noise_variances_df_all = []
    n_df = []
    X_accum_final = []
    Y_accum_final = []

    # Initialize starting points for each repeat.
    for i in range(bo_params['n_repetitions']):

        all_starting_points.append(
            init_points[i][0:bo_params['n_init']])

        if debug_mode_printouts is True:

            message = ('\n\nInit points method ' + str(m) +
                       ',  repetition ' + str(i) + ':\n' +
                       str(all_starting_points[i]))
            print(message)

    for i in indices_of_repeats:

        # Plot the BO for the first five iterations.
        if (i < 5) and (save_figs == True):
            no_plots = False
        else:
            no_plots = True

        if acq_fun_params is None:

            afp = None

        else:

            afp = acq_fun_params.copy()

        if df_data_coll_params is None:

            ddcp = None
            message = 'Start method ' + \
                str(m) + ': No data fusion, repetition ' + str(i)

        else:

            ddcp = df_data_coll_params.copy()
            message = 'Start method ' + \
                str(m) + ': ', ddcp, ', repetition ' + str(i)

        print(message)

        next_suggestions, optimum, model_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, bo_objects = bo_sim_target(
            targetprop_data_source=gt_model_targetprop,
            human_data_source=gt_model_human,
            materials=bo_params['materials'],
            rounds=bo_params['n_rounds'],
            init_points=all_starting_points[i],
            batch_size=bo_params['batch_size'],
            acquisition_function=acquisition_function,
            acq_fun_params=afp,
            df_data_coll_params=ddcp,
            no_plots=no_plots, results_folder=triangle_folder,
            noise_target=bo_params['noise_target'],
            seed=None, close_figs=close_figs)

        if debug_mode_printouts is True:
            # Getting % usage of virtual_memory ( 3rd field)
            print('BO ended. \n')
            print('RAM memory % used:', psutil.virtual_memory()[2])

        # Append results into result variables.
        optima.append(optimum)
        model_optima.append(model_optimum)
        base_model_value = gt_model_targetprop.predict_noiseless(
            model_optimum[:, 0:-1])[0][:, 0]
        regret = np.sqrt(
            np.sum((model_optimum[:, 0:-1] - ground_truth_rA)**2, axis=1))
        regrets.append(regret)
        regret_samples = np.sqrt(
            np.sum((optimum[:, 0:-1] - ground_truth_rA)**2, axis=1))
        regrets_samples.append(regret_samples)
        base_model_values.append(base_model_value)
        X_accum_all.append(X_accum)
        Y_accum_all.append(Y_accum)
        X_accum_final.append(X_accum[-1])
        Y_accum_final.append(Y_accum[-1])
        
        n_df.append(np.zeros((model_optima[i].shape[0],)))
        if data_fusion_params is not None:
            data_fusion_params_all.append(data_fusion_params)
            #X_df_temp = [data_fusion_params['df_data_accum'][i].values[:, 0:-1]
            #             for k in range(bo_params['n_rounds'])]
            X_accum_df_final.append(
                data_fusion_params['df_data_accum'][-1].values[:, 0:-1])
            Y_accum_df_final.append(
                data_fusion_params['df_data_accum'][-1].values[:, [-1]])
            lengthscales_df_all.append(
                np.ravel(data_fusion_params['df_data_hyperpars']['df_data_lengthscales']))
            # variances_df_all.append(data_fusion_params['df_data_hyperpars']['df_data_variances'])
            # gaussian_noise_variances_df_all.append(data_fusion_params['df_data_hyperpars']['df_data_gaussian_noise_variances'])
            df_optima.append(data_fusion_params['df_optimum'])
            df_model_optima.append(data_fusion_params['df_model_optimum'])
            for k in range(bo_params['n_rounds']):
                n_df[i][k] = np.where(~np.isnan(data_fusion_params['df_data_accum'][k].iloc[:,-1]))[0].shape[0]
            
        else:

            X_accum_df_final.append(np.zeros(X_accum_final[i].shape))
            Y_accum_df_final.append(np.zeros(Y_accum_final[i].shape))
            df_optima.append(np.zeros(optima[i].shape))
            df_model_optima.append(np.zeros(model_optima[i].shape))
            
        if X_accum_df_final[i].shape[0] < X_accum_final[i].shape[0]:

            X_accum_df_final[i] = np.concatenate((X_accum_df_final[i],
                                                  np.full(((X_accum_final[i].shape[0] - X_accum_df_final[i].shape[0]), X_accum_df_final[i].shape[1]), np.nan)), axis=0)

        if Y_accum_df_final[i].shape[0] < Y_accum_final[i].shape[0]:

            Y_accum_df_final[i] = np.concatenate((Y_accum_df_final[i],
                                                  np.full(((Y_accum_final[i].shape[0] - Y_accum_df_final[i].shape[0]), Y_accum_df_final[i].shape[1]), np.nan)), axis=0)

        surrogate_model_params_all.append(surrogate_model_params)
        lengthscales_target_all.append(
            np.ravel(surrogate_model_params['lengthscales']))
        variances_target_all.append(surrogate_model_params['variances'])
        gaussian_noise_variances_target_all.append(
            surrogate_model_params['gaussian_noise_variances'])

        if ddcp is None:

            message = 'End method ' + \
                str(m) + ': No data fusion, repetition ' + str(i)

        else:

            message = 'End method ' + \
                str(m) + ': ', ddcp, ', repetition ' + str(i)

        if debug_mode_printouts is True:
            print(message)

        # Example BO objects saved only from the first two repetitions
        # to save disk space.
        if (save_disk_space == False) or (i < 2):

            bo_examples.append([bo_objects])

            filename = modify_filename(pickle_filenames[-1], i+1)

            dbfile = open(filename, 'ab')
            pickle.dump(bo_examples, dbfile)
            dbfile.close()
            outputs["bo_examples"] = bo_examples
            if (save_disk_space == True) and (i == 1):

                # The variable is not needed anymore and it tends to be large so let's delete it.
                del bo_examples

        # Save other results after all repetitions have been done but
        # also 8 times in between if the total number of repetitions
        # is large.
        if (i == (bo_params['n_repetitions']-1)) or (
                (bo_params['n_repetitions'] > 10) and
                (np.remainder((i+1),
                              int(np.floor(bo_params['n_repetitions']/8)))
                 == 0)):

            pickle_variables = ({'optimal_samples': optima,
                                 'model_optima': model_optima},
                                X_accum_all, Y_accum_all,
                                surrogate_model_params_all,
                                data_fusion_params_all)  # , results, lengthscales_all,
            # variances_all, max_gradients_all]

            # Save the results as an backup
            for j in range(len(pickle_variables)):

                # Temporary filename for temp run safe-copies.
                filename = modify_filename(pickle_filenames[j], i+1)

                dbfile = open(filename, 'ab')
                pickle.dump(pickle_variables[j], dbfile)
                dbfile.close()

        if debug_mode_printouts is True:
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            print('Start next repeat...\n')

    print('\nAfter ' + str(indices_of_repeats) +
          'repeats, update outputs for method ' + str(m) + '\n')
    outputs.update(
        dict(
            optima=optima,
            model_optima=model_optima,
            X_accum_final=X_accum_final,
            Y_accum_final=Y_accum_final,
            # surrogate_model_params_all=surrogate_model_params_all,
            base_model_optima=base_model_values,
            regrets=regrets,
            regrets_samples=regrets_samples,
            # data_fusion_params_all = data_fusion_params_all,
            X_accum_df_final=X_accum_df_final,
            Y_accum_df_final=Y_accum_df_final,
            df_optima=df_optima,
            df_model_optima=df_model_optima,
            # lengthscales_df_all = lengthscales_df_all,
            # variances_df_all = variances_df_all,
            # gaussian_noise_variances_df_all = gaussian_noise_variances_df_all,
            lengthscales_target_all=lengthscales_target_all,
            variances_target_all=variances_target_all,
            gaussian_noise_variances_target_all=gaussian_noise_variances_target_all,
            n_df = n_df
        )
    )
    if debug_mode_printouts is True:
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, '\n')

    return outputs


def n_samples(n_init, n_rounds, batch_size):

    n_samples = n_init + (n_rounds-1)*batch_size

    return n_samples


def calc_additional_idx_for_folder(param1, param2):

    param1_array = cg(np.linspace(0.01, 1, 8))
    param2_array = np.linspace(0, 1, 6)

    idx = (np.argwhere(param1 == param1_array)*len(param2_array) +
           np.argwhere(param2 == param2_array))[0, 0]

    return idx


###############################################################################
# Create settings
settings = pyqsl.Settings()
settings.path_gtmodel_targetprop = './Source_data/gt_model_target_variable_equal_AB'
settings.path_gtmodel_humanevals = './Source_data/gt_model_support_variable_0mean_1std'
settings.init_points = np.array(np.load('./Source_data/initpts_longer.npy'))
# settings.n_init_points = 10
settings.gt_model_targetprop = pyqsl.Setting(relation=pyqsl.Function(
    function=load_GP_model, parameters={"path_model": "path_gtmodel_targetprop"}))
settings.gt_model_human = pyqsl.Setting(relation=pyqsl.Function(
    function=load_GP_model, parameters={"path_model": "path_gtmodel_humanevals"}))
settings.folder = './Results/20241018/eAB-dfhyperpars/Noise0/beta-vs-01/'
settings.additional_idx_for_folder = None
settings.c_eig = 0.6000000000000001  # 0.1
settings.c_exclz = 1
settings.c_g = 0.15142857 #0.178  # cg(np.array([0.9]))
settings.jitter = 2  # 7
settings.n_repetitions = 50
settings.n_rounds = 25
settings.n_init = 3
settings.batch_size = 1
settings.materials = ['CsPbI', 'MAPbI', 'FAPbI']
settings.mat_dim = ['CsPbI', 'MAPbI', 'FAPbI', 'target']
settings.target_dim = ['target']
settings.noise_target = 0
settings.noise_df = pyqsl.Setting(relation='noise_target')
settings.save_figs = False
settings.save_disk_space = True
settings.close_figs = True
settings.ground_truth_rA = np.array([[0.165, 0.035, 0.795]])
settings.indices_of_repeats = list(range(settings.n_repetitions.value))
settings.acquisition_function = 'LCB_DF'
# pyqsl.Setting(relation=pyqsl.Function(function=repeats_to_range))
settings.rounds_as_list = list(range(settings.n_rounds.value))
settings.n_samples = n_samples(settings.n_init.value, settings.n_rounds.value,
                               settings.batch_size.value)  # pyqsl.Setting(relation=pyqsl.Function(function=n_samples, parameters={"n_init":"n_init", "n_rounds": "n_rounds", "batch_size": "batch_size"}))
settings.samples_as_list = [i for i in range(settings.n_samples.value)]
# pyqsl.Setting(relation=pyqsl.Function(function=repeats_to_range, parameters={"n_repetitions": "n_rounds"}))
settings.bo_params = pyqsl.Setting(relation=pyqsl.Function(function=bo_args_to_dict, parameters={
    "n_repetitions": "n_repetitions",
    "n_rounds": "n_rounds",
    "n_init": "n_init",
    "batch_size": "batch_size",
    "materials": "materials",
    "noise_target": "noise_target",
    "acquisition_function": "acquisition_function"
}))
settings.m = 2
settings.optima = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list", "mat_dim"])
settings.model_optima = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list", "mat_dim"])
settings.df_optima = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list", "mat_dim"])
settings.df_model_optima = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list", "mat_dim"])
settings.base_model_optima = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.regrets = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.regrets_samples = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.X_accum_final = pyqsl.Setting(
    dimensions=["indices_of_repeats", "samples_as_list", "materials"])
settings.Y_accum_final = pyqsl.Setting(
    dimensions=["indices_of_repeats", "samples_as_list", "target_dim"])
settings.X_accum_df_final = pyqsl.Setting(
    dimensions=["indices_of_repeats", "samples_as_list", "materials"])
settings.Y_accum_df_final = pyqsl.Setting(
    dimensions=["indices_of_repeats", "samples_as_list", "target_dim"])
# settings.lengthscales_df_all = pyqsl.Setting(dimensions=["indices_of_repeats", "rounds_as_list"])
# settings.variances_df_all = pyqsl.Setting(dimensions=["indices_of_repeats", "rounds_as_list"])
# settings.gaussian_noise_variances_df_all = pyqsl.Setting(dimensions=["indices_of_repeats", "rounds_as_list"])
settings.lengthscales_target_all = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.variances_target_all = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.gaussian_noise_variances_target_all = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])
settings.n_df = pyqsl.Setting(
    dimensions=["indices_of_repeats", "rounds_as_list"])

###############################################################################
# SWEEP OVER THESE PARAMS
#c_g_options = cg(np.linspace(0.01, 1, 8))
#c_eig_options = np.linspace(0, 1, 6)

# Set relations
# settings.jitter.relation = pyqsl.Equation(equation="4 + noise_target * 5")
#settings.additional_idx_for_folder.relation = pyqsl.Function(function=calc_additional_idx_for_folder, parameters={
#    "param1": "c_g",
#    "param2": "c_eig"
#})

#settings.batch_size.relation = pyqsl.Equation(equation="1 + noise_target * 2")
#settings.additional_idx_for_folder.relation = pyqsl.Function(function=calc_additional_idx_for_folder, parameters={
#    "param1": "c_g",
#    "param2": "c_eig"
#})

# Execute task

# Sweep over exploration parameter.
result = pyqsl.run(task=task, settings=settings, sweeps=dict(jitter=np.linspace(0,7,8),
                                                             noise_target=np.linspace(0, 1, 3),
    m=np.array([0,1])
    #c_eig=c_eig_options,
    #c_g=c_g_options
),
    parallelize=True
)


# Sweep over noise level.
# result = pyqsl.run(task=task, settings=settings, sweeps=dict(noise_target=np.linspace(0, 1, 10)), parallelize=True)

result.save(settings.folder.value + '/' + 'results.pickle')

# Plots for optima
noise_target = 1
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', noise_target=noise_target, drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', noise_target=noise_target, drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.base_model_optima.mean(dim='indices_of_repeats').sel(
    noise_target=noise_target, drop=True).plot.line(x="rounds_as_list")
plt.show()

##
# Comparison of optima types plot
result.dataset.base_model_optima.mean(dim='indices_of_repeats').isel(
    rounds_as_list=-1).sel(noise_target=noise_target, drop=True).plot(label='base_model_optima')
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', noise_target=noise_target, drop=True).isel(rounds_as_list=-1).plot(label='optima')
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', noise_target=noise_target, drop=True).isel(rounds_as_list=-1).plot(label='model_optima')
plt.legend()
plt.show()

# Regret plot
result.dataset.regrets.mean(dim='indices_of_repeats').sel(
    noise_target=noise_target, drop=True).plot.line(x='rounds_as_list')
plt.show()

# Scatter plot on optima
myx = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(noise_target=noise_target, mat_dim='CsPbI', drop=True)
myy = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(noise_target=noise_target, mat_dim='MAPbI', drop=True)
plt.figure()
for i in range(2):
    # 'Jitter #' + str(i))
    plt.scatter(myx[i, :], myy[i, :], label='Method ' + str(i))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend()
plt.show()


# Scatter plot on human evals
noise_target = 1.0
m = 0
myx = result.dataset.X_accum_df_final.sel(
    materials='CsPbI', noise_target=noise_target, m=m, drop=True)
myy = result.dataset.X_accum_df_final.sel(
    materials='MAPbI', noise_target=noise_target, m=m, drop=True)
myz = result.dataset.Y_accum_df_final.sel(
    target_dim='target', noise_target=noise_target, m=m, drop=True)
plt.figure()
for i in range(50):
    # , label = 'Method ' + str(i))#'Jitter #' + str(i))
    plt.scatter(myx[i, :], myy[i, :], c=myz[i, :])
plt.xlim((0, 1))
plt.ylim((0, 1))
# plt.legend()
plt.show()

###############################################################################
# M0 M1 COMPARISONS

# Plots for optima
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.base_model_optima.mean(
    dim='indices_of_repeats').plot.line(x="rounds_as_list")
plt.show()

##
# Comparison of optima types plot
result.dataset.base_model_optima.mean(dim='indices_of_repeats').isel(
    rounds_as_list=-1).plot(label='base_model_optima')
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot(label='optima')
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot(label='model_optima')
plt.legend()
plt.show()

# Regret plot
result.dataset.regrets.mean(
    dim='indices_of_repeats').plot.line(x='rounds_as_list')
plt.show()

result.dataset.regrets_samples.mean(
    dim='indices_of_repeats').plot.line(x='rounds_as_list')
plt.show()

# Scatter plot on optima
myx = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='CsPbI', drop=True)
myy = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='MAPbI', drop=True)
myz = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='target', drop=True)
plt.figure()
for i in range(3):
    # 'Jitter #' + str(i))
    plt.scatter(myx[i, :], myy[i, :], label='Method ' + str(i))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend()
plt.show()

# The same as above but separate figs and color is the value of the optimum.
for i in range(3):
    plt.figure()
    plt.scatter(myx[i, :], myy[i, :], c=myz[i, :],
                label='Method ' + str(i))  # 'Jitter #' + str(i))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.colorbar()
    plt.legend()
    plt.show()


# Scatter plot on human evals
# About 90% of samples should be below this limit for 0mean 1 std normal distribution
noise_target = 1.4
m = 1
myx = result.dataset.X_accum_df_final.sel(materials='CsPbI', m=m, drop=True)
myy = result.dataset.X_accum_df_final.sel(materials='MAPbI', m=m, drop=True)
myz = result.dataset.Y_accum_df_final.sel(target_dim='target', m=m, drop=True)
plt.figure()
# for i in range(5):
# , label = 'Method ' + str(i))#'Jitter #' + str(i))
plt.scatter(np.ravel(myx), np.ravel(myy), c=np.ravel(myz))
plt.xlim((0, 1))
plt.ylim((0, 1))
# plt.legend()
plt.show()


###############################################################################

# 2D plots
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.base_model_optima.mean(
    dim='indices_of_repeats').isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.regrets.mean(dim='indices_of_repeats').isel(
    rounds_as_list=16).plot()
plt.show()


# Compare benchmark criteria on fixed jitter, varying noise.
jitter = 9.25
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', jitter=jitter, drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', jitter=jitter, drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.base_model_optima.mean(dim='indices_of_repeats').sel(
    jitter=jitter, drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.regrets.mean(dim='indices_of_repeats').sel(
    jitter=jitter, drop=True).plot.line(x='rounds_as_list')
plt.show()


# 2D plots comparing methods
for i in range(2):
    result.dataset.optima.mean(dim='indices_of_repeats').sel(
        mat_dim='target', m=i, drop=True).isel(rounds_as_list=-1).plot()
plt.show()
for i in range(2):
    result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
        mat_dim='target', m=i, drop=True).isel(rounds_as_list=-1).plot()
plt.show()
for i in range(2):
    result.dataset.base_model_optima.mean(dim='indices_of_repeats').sel(
        m=i, drop=True).isel(rounds_as_list=-1).plot()
plt.show()
for i in range(2):
    result.dataset.regrets.mean(dim='indices_of_repeats').sel(
        m=i, drop=True).isel(rounds_as_list=-1).plot()
plt.show()

# Basic plots with 2D runs.
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.base_model_optima.mean(
    dim='indices_of_repeats').plot.line(x="rounds_as_list")
plt.show()
# Regret plot
result.dataset.regrets.mean(
    dim='indices_of_repeats').plot.line(x='rounds_as_list')
plt.show()


# DF EIG HO

# Compare benchmark criteria on fixed jitter, varying noise.
# jitter = 9.25
for i in range(len(c_g_options)):
    result.dataset.optima.mean(dim='indices_of_repeats').sel(
        mat_dim='target', c_g=c_g_options[i], drop=True).plot.line(x="rounds_as_list")
plt.show()
for i in range(len(c_g_options)):
    result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
        mat_dim='target', c_g=c_g_options[i], drop=True).plot.line(x="rounds_as_list")
plt.show()
for i in range(len(c_g_options)):
    result.dataset.base_model_optima.mean(dim='indices_of_repeats').sel(
        c_g=c_g_options[i], drop=True).plot.line(x="rounds_as_list")
plt.show()
for i in range(len(c_g_options)):
    result.dataset.regrets.mean(dim='indices_of_repeats').sel(
        c_g=c_g_options[i], drop=True).plot.line(x='rounds_as_list')
plt.show()
for i in range(len(c_g_options)):
    result.dataset.regrets_samples.mean(dim='indices_of_repeats').sel(
        c_g=c_g_options[i], drop=True).plot.line(x='rounds_as_list')
plt.show()


result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.base_model_optima.mean(
    dim='indices_of_repeats').isel(rounds_as_list=-1).plot()
plt.show()
result.dataset.regrets.mean(dim='indices_of_repeats').isel(
    rounds_as_list=-1).plot()
plt.show()
result.dataset.regrets_samples.mean(dim='indices_of_repeats').isel(
    rounds_as_list=16).plot()
plt.show()

# Joint scatter plots on model optima
round_to_plot = -1
plt.figure()
plt.title('All c_eig, all_repetitions, round=' + str(round_to_plot))
plt.xlabel('Cs proportion')
plt.ylabel('MA proportion')
for i in range(1,2):#len(c_g_options)):
    myx = result.dataset.model_optima.isel(
        rounds_as_list=round_to_plot, c_g=i).sel(mat_dim='CsPbI', drop=True)
    myy = result.dataset.model_optima.isel(
        rounds_as_list=round_to_plot, c_g=i).sel(mat_dim='MAPbI', drop=True)
    myz = result.dataset.model_optima.isel(
        rounds_as_list=round_to_plot, c_g=i).sel(mat_dim='target', drop=True)
    plt.scatter(np.ravel(myx), np.ravel(myy), label='c_g = ' +
            str(c_g_options[i]))  # 'Jitter #' + str(i))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend()
plt.show()

# Sanity check: The values of the base model optima should be exactly the
# same than sample optima in a noise-free case if everything is alright.
# Scatter plot on base model optima
round_to_plot = -1
plt.figure()
plt.title('All c_eig, all_repetitions, round=' + str(round_to_plot))
plt.xlabel('Cs proportion')
plt.ylabel('MA proportion')
myx = result.dataset.model_optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='CsPbI', drop=True)
myy = result.dataset.model_optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='MAPbI', drop=True)
myz = result.dataset.model_optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='target', drop=True)
plt.scatter(np.ravel(myx), np.ravel(myy), c = np.ravel(myz))  # 'Jitter #' + str(i))
plt.xlim((0, 1))
plt.ylim((0, 1))
cbar = plt.colorbar()
cbar.set_label('Target value')
plt.show()

round_to_plot = -1
plt.figure()
plt.title('All c_eig, all_repetitions, round=' + str(round_to_plot))
plt.xlabel('Cs proportion')
plt.ylabel('MA proportion')
myx = result.dataset.optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='CsPbI', drop=True)
myy = result.dataset.optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='MAPbI', drop=True)
myz = result.dataset.optima.isel(
    rounds_as_list=round_to_plot).sel(mat_dim='target', drop=True)
plt.scatter(np.ravel(myx), np.ravel(myy), c = np.ravel(myz))  # 'Jitter #' + str(i))
plt.xlim((0, 1))
plt.ylim((0, 1))
cbar = plt.colorbar()
cbar.set_label('Target value')
plt.show()


# Scatter plot on human evals
noise_target = 1.0
m = 0
myx = result.dataset.X_accum_df_final.sel(
    materials='CsPbI', noise_target=noise_target, m=m, drop=True)
myy = result.dataset.X_accum_df_final.sel(
    materials='MAPbI', noise_target=noise_target, m=m, drop=True)
myz = result.dataset.Y_accum_df_final.sel(
    target_dim='target', noise_target=noise_target, m=m, drop=True)
plt.figure()
for i in range(50):
    # , label = 'Method ' + str(i))#'Jitter #' + str(i))
    plt.scatter(myx[i, :], myy[i, :], c=myz[i, :])
plt.xlim((0, 1))
plt.ylim((0, 1))
# plt.legend()´
plt.show()

'''
# Joint scatter plots on sample optima
for i in range(len(c_g_options)):
    plt.figure()
    title = 'c_g=' + str(c_g_options[i])
    myx = result.dataset.optima.isel(
        rounds_as_list=-1, c_g=i).sel(mat_dim='CsPbI', drop=True)
    myy = result.dataset.optima.isel(
        rounds_as_list=-1, c_g=i).sel(mat_dim='MAPbI', drop=True)
    myz = result.dataset.optima.isel(
        rounds_as_list=-1, c_g=i).sel(mat_dim='target', drop=True)
    plt.scatter(myx[i, :], myy[i, :], label='c_g = ' +
                str(c_g_options[i]))  # 'Jitter #' + str(i))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend()
    plt.show()

# Scatter plot on model optima
for i in range(len(c_g_options)):
    for j in range(len(c_eig_options)):
        plt.figure()
        title = 'c_g=' + str(c_g_options[i]) + \
            ', c_eig=' + str(c_eig_options[j])
        myx = result.dataset.model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='CsPbI', drop=True)
        myy = result.dataset.model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='MAPbI', drop=True)
        myz = result.dataset.model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=i).sel(mat_dim='target', drop=True)
        # , label = 'c_eig = ' + str(c_eig_options[i]))#'Jitter #' + str(i))
        plt.scatter(myx, myy, c=myz)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.colorbar()
        plt.legend()
        plt.show()

# Scatter plot on sample optima
for i in range(len(c_g_options)):
    for j in range(len(c_eig_options)):
        plt.figure()
        title = 'c_g=' + str(c_g_options[i]) + \
            ', c_eig=' + str(c_eig_options[j])
        myx = result.dataset.optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='CsPbI', drop=True)
        myy = result.dataset.optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='MAPbI', drop=True)
        myz = result.dataset.optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=i).sel(mat_dim='target', drop=True)
        # , label = 'c_eig = ' + str(c_eig_options[i]))#'Jitter #' + str(i))
        plt.scatter(myx, myy, c=myz)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.colorbar()
        plt.legend()
        plt.show()

# Scatter plot on base model optima
for i in range(len(c_g_options)):
    for j in range(len(c_eig_options)):
        plt.figure()
        title = 'c_g=' + str(c_g_options[i]) + \
            ', c_eig=' + str(c_eig_options[j])
        myx = result.dataset.base_model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='CsPbI', drop=True)
        myy = result.dataset.base_model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=j).sel(mat_dim='MAPbI', drop=True)
        myz = result.dataset.base_model_optima.isel(
            rounds_as_list=-1, c_g=i, c_eig=i).sel(mat_dim='target', drop=True)
        # , label = 'c_eig = ' + str(c_eig_options[i]))#'Jitter #' + str(i))
        plt.scatter(myx, myy, c=myz)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.colorbar()
        plt.legend()
        plt.show()

'''



# SINGLE RUN

# Compare benchmark criteria on fixed jitter, varying noise.
result.dataset.optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
    mat_dim='target', drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.base_model_optima.mean(dim='indices_of_repeats').sel(
    drop=True).plot.line(x="rounds_as_list")
plt.show()
result.dataset.regrets.mean(dim='indices_of_repeats').sel(
    drop=True).plot.line(x='rounds_as_list')
plt.show()


print(result.dataset.optima.mean(dim='indices_of_repeats').sel(mat_dim='target', drop=True).isel(rounds_as_list=-1).values,
      result.dataset.model_optima.mean(dim='indices_of_repeats').sel(
          mat_dim='target', drop=True).isel(rounds_as_list=-1).values,
      result.dataset.base_model_optima.mean(
          dim='indices_of_repeats').isel(rounds_as_list=-1).values,
      result.dataset.regrets.mean(dim='indices_of_repeats').isel(rounds_as_list=-1).values)

# Scatter plot on model optima
plt.figure()
myx = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='CsPbI', drop=True)
myy = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='MAPbI', drop=True)
myz = result.dataset.model_optima.isel(
    rounds_as_list=-1).sel(mat_dim='target', drop=True)
# , label = 'c_eig = ' + str(c_eig_options[i]))#'Jitter #' + str(i))
plt.scatter(myx, myy, c=myz)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.colorbar()
plt.legend()
plt.show()

# Scatter plot on optima
plt.figure()
title = 'c_g=' + str(c_g_options[i]) + ', c_eig=' + str(c_eig_options[j])
myx = result.dataset.optima.isel(
    rounds_as_list=-1).sel(mat_dim='CsPbI', drop=True)
myy = result.dataset.optima.isel(
    rounds_as_list=-1).sel(mat_dim='MAPbI', drop=True)
myz = result.dataset.optima.isel(
    rounds_as_list=-1).sel(mat_dim='target', drop=True)
# , label = 'c_eig = ' + str(c_eig_options[i]))#'Jitter #' + str(i))
plt.scatter(myx, myy, c=myz)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.colorbar()
plt.legend()
plt.show()


# Scatter plot on human evals
noise_target = 1.0
m = 0
myx = result.dataset.X_accum_df_final.sel(
    materials='CsPbI', noise_target=noise_target, m=m, drop=True)
myy = result.dataset.X_accum_df_final.sel(
    materials='MAPbI', noise_target=noise_target, m=m, drop=True)
myz = result.dataset.Y_accum_df_final.sel(
    target_dim='target', noise_target=noise_target, m=m, drop=True)
plt.figure()
for i in range(50):
    # , label = 'Method ' + str(i))#'Jitter #' + str(i))
    plt.scatter(myx[i, :], myy[i, :], c=myz[i, :])
plt.xlim((0, 1))
plt.ylim((0, 1))
# plt.legend()
plt.show()
