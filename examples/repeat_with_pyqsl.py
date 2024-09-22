import matplotlib.pyplot as plt
import numpy as np
import pyqsl
from hper_util_gp import load_GP_model
from hper_util_repetitions_lcb import cg, build_filenames, set_bo_settings, set_repeat_settings, modify_filename
from hper_bo import bo_sim_target
from set_figure_defaults import FigureDefaults
import pickle
#
fig_def = FigureDefaults(style='presentation')

## Task function

def bo_args_to_dict(n_repetitions, n_rounds, n_init, batch_size, materials, noise_target):
    return {
    "n_repetitions": n_repetitions,
    "n_rounds": n_rounds,
    "n_init": n_init,
    "batch_size": batch_size,
    "materials": materials,
    "noise_target": noise_target,
    }

def repeats_to_range(n_repetitions):
    return list(range(n_repetitions))

def task(m, c_g, c_exclz, c_eig, jitters, bo_params, noise_df, folder, init_points, indices_of_repeats, save_figs,
         gt_model_targetprop, gt_model_human, close_figs, save_disk_space,
         ):
    data_fusion_property, df_data_coll_method, acquisition_function, c_grad, c_e, jitter, fetch_file_date = set_repeat_settings(m, [c_g], [c_exclz], [c_eig], [jitters])
    outputs = {}
    if (m > -1):
        ###############
        # Typically, one does not need to modify these inputs.
        acq_fun_descr, acq_fun_params, df_data_coll_descr, df_data_coll_params = set_bo_settings(
            bo_params, acquisition_function, jitter, data_fusion_property, 
            df_data_coll_method, noise_df, c_grad, c_e)
        
        # Create result folders and build filenames for result files.
        pickle_filenames, figure_filenames, triangle_folder = build_filenames(
            folder, bo_params, acq_fun_descr, df_data_coll_descr,
            fetch_file_date=fetch_file_date, m=m)
        
        ###############

        all_starting_points = []
        bo_examples = []
        optima = []
        model_optima = []
        X_accum_all = []
        Y_accum_all = []
        data_fusion_params_all = []
        surrogate_model_params_all =  []
        
        # Initialize starting points for each repeat.
        for i in range(bo_params['n_repetitions']):

            all_starting_points.append(
                init_points[i][0:bo_params['n_init']])

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
                message = 'Start method ' + str(m) + ': No data fusion, repetition ' + str(i)
                
            else:

                ddcp = df_data_coll_params.copy()
                message = 'Start method ' + str(m) + ': ', ddcp, ', repetition ' + str(i)
                

            print(message)
                            
            next_suggestions, optimum, model_optimum, X_rounds, Y_rounds, X_accum, Y_accum, surrogate_model_params, data_fusion_params, bo_objects = bo_sim_target(
                targetprop_data_source = gt_model_targetprop,
                human_data_source = gt_model_human,
                materials=bo_params['materials'],
                rounds=bo_params['n_rounds'],
                init_points=all_starting_points[i],
                batch_size=bo_params['batch_size'],
                acquisition_function=acquisition_function,
                acq_fun_params=afp,
                df_data_coll_params=ddcp,
                no_plots=no_plots, results_folder=triangle_folder,
                noise_target = bo_params['noise_target'],
                seed = None, close_figs = close_figs)
            
            # Getting % usage of virtual_memory ( 3rd field)
            print('BO ended. \n')
            print('RAM memory % used:', psutil.virtual_memory()[2])
            
            
            optima.append(optimum)
            model_optima.append(model_optimum)
            X_accum_all.append(X_accum)
            Y_accum_all.append(Y_accum)

            if data_fusion_params is not None:
                data_fusion_params_all.append(data_fusion_params)
            
            surrogate_model_params_all.append(surrogate_model_params)
            
            if ddcp is None:
                
                message = 'End method ' + str(m) + ': No data fusion, repetition ' + str(i)
                
            else:
                
                message = 'End method ' + str(m) + ': ', ddcp, ', repetition ' + str(i)

            print(message)
            
            # Example BO objects saved only from the first two repetitions
            # to save disk space.
            if (save_disk_space is False) or (i < 2):
                
                bo_examples.append([bo_objects])
                
                filename = modify_filename(pickle_filenames[-1], i+1)

                dbfile = open(filename, 'ab')
                pickle.dump(bo_examples, dbfile)
                dbfile.close()
                outputs["bo_examples"] = bo_examples
                if (save_disk_space is True) and (i == 1):
                    
                    # The variable is not needed anymore and it tends to be large so let's delete it.
                    del bo_examples
                    
            # Save other results after all repetitions have been done but
            # also  times in between if the total number of repetitions
            # is large.
            if (i == (bo_params['n_repetitions']-1)) or (
                    (bo_params['n_repetitions'] > 10) and
                    (np.remainder((i+1),
                                  int(np.floor(bo_params['n_repetitions']/10)))
                     == 0)):

                pickle_variables = ({'optimal_samples': optima,
                                     'model_optima': model_optima}, 
                                    X_accum_all, Y_accum_all,
                                    surrogate_model_params_all, 
                                    data_fusion_params_all) #, results, lengthscales_all,
                                    #variances_all, max_gradients_all]
                                    
                # Save the results as an backup
                for j in range(len(pickle_variables)):
                    
                    # Temporary filename for temp run safe-copies.
                    filename = modify_filename(pickle_filenames[j], i+1)

                    dbfile = open(filename, 'ab')
                    pickle.dump(pickle_variables[j], dbfile)
                    dbfile.close()
                    
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            print('Start next repeat...\n')
        outputs.update(
            dict(
                optima=optima,
                model_optima=model_optima,
                X_accum_all = X_accum_all,
                Y_accum_all = Y_accum_all,
                surrogate_model_params_all=surrogate_model_params_all,
                data_fusion_params_all=data_fusion_params_all,
            )
        )
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, '\n')
        print('Clearing variables...\n')
    return outputs
        
## Create settings


settings = pyqsl.Settings()
settings.path_gtmodel_targetprop = './Source_data/gt_model_target_variable_edges'
settings.path_gtmodel_humanevals = './Source_data/visualquality/human_model_scale0to1'
settings.init_points = np.array(np.load('./Source_data/initpts.npy'))
settings.n_init_points = 10
settings.gt_model_targetprop = pyqsl.Setting(relation=pyqsl.Function(function=load_GP_model, parameters={"path_model":"path_gtmodel_targetprop"}))
settings.gt_model_human = pyqsl.Setting(relation=pyqsl.Function(function=load_GP_model, parameters={"path_model":"path_gtmodel_humanevals"}))
settings.folder = './Results/20240920/LCB_DF/Test3/Noise000/'
settings.c_eig = 0.1
settings.c_exclz = 1
settings.c_g = cg(np.array([0.9]))
settings.jitters = 5
settings.n_repetitions = 15
settings.n_rounds = 15
settings.n_init = 1
settings.batch_size = 1
settings.materials = ['CsPbI', 'MAPbI', 'FAPbI']
settings.mat_dim = ['CsPbI', 'MAPbI', 'FAPbI', 'target']
settings.noise_target = 1.0
settings.noise_df = pyqsl.Setting(relation='noise_target')
settings.save_figs = False
settings.save_disk_space = False
settings.close_figs = True
settings.indices_of_repeats = list(range(settings.n_repetitions.value))
#pyqsl.Setting(relation=pyqsl.Function(function=repeats_to_range))
settings.rounds_as_list = list(range(settings.n_rounds.value))

#pyqsl.Setting(relation=pyqsl.Function(function=repeats_to_range, parameters={"n_repetitions": "n_rounds"}))
settings.bo_params = pyqsl.Setting(relation=pyqsl.Function(function=bo_args_to_dict, parameters={
    "n_repetitions": "n_repetitions",
    "n_rounds": "n_rounds",
    "n_init": "n_init",
    "batch_size": "batch_size",
    "materials": "materials",
    "noise_target": "noise_target",
}))
settings.m = 0
settings.optima = pyqsl.Setting(dimensions=["indices_of_repeats", "rounds_as_list", "mat_dim"])

## Execute task
result = pyqsl.run(task=task, settings=settings, sweeps=dict(noise_target=np.linspace(0, 1, 5)))
result.dataset.optima.mean(dim='indices_of_repeats').sel(mat_dim='target', drop=True).plot.line(x="rounds_as_list")
