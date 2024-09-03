import pickle
import numpy as np
import pandas as pd
import os

def read_pickle(filename):
    
    try:
        
        with open(filename,'rb') as dbfile:
            
            ss = pickle.load(dbfile)
            dbfile.close()
                
    except IOError:
        
        raise Exception('No file was found. Check the filename: ' + filename)
    
    return ss

def save_pickle(filename, data):
    
    with open(filename,'ab') as dbfile:
        
        pickle.dump(data, dbfile)
        dbfile.close()
    

folder_target = './$WRKDIR/Results/20240902/Test_ho_long/4/'
folder_source = './$WRKDIR/Results/20240902/Test_ho_long/Merged3/'
res_all_folder = './$WRKDIR/Results/20240902/Test_ho_long/Merged4/'
n_repetitions_target = 25
n_repetitions_source = 18
diff = -2

###############################################################################

n_reps_true = n_repetitions_target + diff


if not os.path.exists(res_all_folder):
    os.makedirs(res_all_folder)

# Assumes the folders have been created in the time order.
folder_list_target = np.sort((next(os.walk(folder_target), (None, [], None)))[1])
folder_list_source = np.sort((next(os.walk(folder_source), (None, [], None)))[1])

for m in range(len(folder_list_target)):
    
    res_folder = res_all_folder + folder_list_target[m] + '/'
    
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    # All files in the result folder in question.
    file_list_target = np.sort((next(os.walk(folder_target + folder_list_target[m]), (None, [], None)))[2])
    file_list_source = np.sort((next(os.walk(folder_source + folder_list_source[m]), (None, [], None)))[2])
    
    for i in range(len(file_list_target)):
        
        t = file_list_target[i]
        
        # Merge only the files with desired number of repetitions.
        if t.find('_nreps' + str(n_repetitions_target))>-1:
            
            
            
            # Target file:
            # Prefix end index for filenames.
            idx0_target = t.find('_')
            prefix_target = t[0:idx0_target]
            # Nreps start index for filenames.
            idx1_target = t.find('_nreps') + 6
            # Nreps end index for filenames.
            idx2_target = t.find('_', idx1_target + 1)
            
            # The corresponding source file is otherwise the same but prefix 
            # and the number of repetitions are different:
            
            # Prefix end index for filenames.
            idx0_source = file_list_source[0].find('_')
            prefix_source = file_list_source[0][0:idx0_source]
            
            s = prefix_source + t[idx0_target:idx1_target] + str(n_repetitions_source) + t[idx2_target::]
            
            # Add the full paths.
            t_path = folder_target + folder_list_target[m] + '/' + t
            s_path = folder_source + folder_list_source[m] + '/' + s
            
            # The corresponding merged target file will be saved as:
            mt = prefix_target + t[idx0_target:idx1_target] + str(n_reps_true) + t[idx2_target::]
            mt_path = res_folder + mt
            
            # Fetch the results from pickled backup
            data_t = read_pickle(t_path)
            data_s = read_pickle(s_path)
            
            print('Sum of the target and source lengths for file ' + 
                  t + ':\n' + str(len(data_t)) + ' + ' + str(len(data_s)))
            
            if (len(data_t) + len(data_s)) == n_reps_true:
                
                data_t.extend(data_s)
                save_pickle(mt_path, data_t)
                
            elif t.find('optima')>-1:
                
                # This variable is a dictionary with a fixed number of keys.
                # Merge key contents instead.
                print('Sum of the target and source lengths for contents of file ' + 
                      t + ':\n' + str(len(data_t['optimal_samples'])) + 
                      ' + ' + str(len(data_s['model_optima'])))
                
                if (len(data_t['optimal_samples']) + 
                    len(data_s['optimal_samples'])) == n_reps_true:
                    
                    data_t['optimal_samples'].extend(data_s['optimal_samples'])
                    data_t['model_optima'].extend(data_s['model_optima'])
                    save_pickle(mt_path, data_t)
                    
            elif t.find('data_fusion_params')>-1:
                
                # This variable is empty for vanilla BO (with no data fusion).
                if (len(data_t) + len(data_s)) == 0:
                    
                    save_pickle(mt_path, data_t)
                    
            else:
                
                # Some other file type that this script does not work for. Do
                # nothing.
                print('The target number of repetitions is ' + 
                                str(n_repetitions_target) + 
                                ' but the sum of repetitions in the files is ' + 
                                str(len(data_s)) + ' + ' + str(len(data_t)) + 
                                '. These were not merged. Check the files first.')
                
                
     