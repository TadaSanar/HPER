#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:49:57 2023

@author: armi
"""

from GPyOpt.acquisitions import AcquisitionEI_DFT
from GPyOpt.models import GPModel
materials=['CsPbI', 'MAPbI', 'FAPbI']
domain_boundaries = [0, 1]

bounds = []
for j in range(len(materials)):
    bounds.append({'name': materials[j], 'type': 'continuous',
                   'domain': domain_boundaries})
    
ei_dft_params = {'df_data': None,
             'df_target_var': 'Yellowness',
             'df_input_var': ['CsPbI', 'MAPbI', 'FAPbI'],
             'gp_lengthscale': 0.03,
             'gp_variance': 2,
             'p_beta': 0.025,
             'p_midpoint': 0
             }

eidft = AcquisitionEI_DFT(GPModel(), bounds,'lbfgs', None, 0.1, ei_dft_params)






c_eig = [0.1, 0.5] # Expected information gain.
c_exclz = [5,10,15] # Size of the exclusion zone in percentage points (max. 100)
c_g = [0.1, 0.6] # Gradient limit. 0.05#, 0.07, 0.1, 0.2, 0.5, 0.75

hyperparams_eig = []
hyperparams_exclz = []
for i in range(len(c_g)):
    for j in range(len(c_exclz)):

        hyperparams_exclz.append((c_g[i], c_exclz[j]))

    for j in range(len(c_eig)):

        hyperparams_eig.append((c_g[i], c_eig[j]))
        
jitters = [0.01, 0.02, 0.05, 0.1]


n_eig  = len(hyperparams_eig)
n_exclz  = len(hyperparams_exclz)
n_hpars = 2 + n_eig + n_exclz
n_j = len(jitters)


for m in range(48):

    if (m % n_hpars) == 0:

        print('Vanilla, ', m, m // n_hpars)
        
    elif (m % n_hpars) == 1:

        print('Always, ', m, m // n_hpars)
        
    elif (m % n_hpars) < (n_hpars - n_exclz):
        
        c_grad = hyperparams_eig[(m % n_hpars)-2][0]
        c_e = hyperparams_eig[(m % n_hpars)-2][1]
        print('EIG, ', m, c_grad, c_e, m // n_hpars)
        
    else:
        
        c_grad = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][0]
        c_e = hyperparams_exclz[(m % n_hpars) - (n_hpars - n_exclz)][1]
        print('Exclz, ', m, c_grad, c_e, m // n_hpars)
        