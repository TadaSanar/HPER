#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:50:15 2024

@author: atiihone

Utility script for determining suitable parameters for the inverted sigmoid
function that defines the probability disctribution scaling.

"""

import numpy as np
import matplotlib.pyplot as plt

def inv_sigmoid(mean, midpoint, beta):
    
    # Inverted because the negative Gibbs energies are the ones that are stable.
    f = 1/(1+np.exp((mean-midpoint)/beta))
    
    return f

###############################################################################
# SET PARAMETERS

# Code implementation assumes human evaluations are given with a numeric range
# of [0,1] so that lower values mean BETTER quality samples. Please provide
# below two estimates on how the probability distribution looks in your
# application (based on your preferences and domain knowledge).
 
# Which numeric sample quality rating will you give when half of the
# synthesized samples are of high quality?
# Default value is 0.66 (explanation: we evaluated sample quality using a 
# 4-level rating of {0, 0.33, 0.66, 1}).
x_50 = 0.5#0.5

# Which numeric sample quality rating will you give when 80% of the
# synthesized samples are of high quality?
# Default value is 0.5 (explanation: we evaluated sample quality using a 
# 4-level rating of {0, 0.33, 0.66, 1} and wanted the algorithm to remain
# conservative regarding ruling any samples fully out from the search).
x_80 = 0.2#0.33

###############################################################################
# NO CHANGES NEEDED
# Computing the hyperparameter values for the sigmoid function that defines the
# probability distribution.

# Range of human evaluation values.
x = np.arange(-1,1.7,0.05)

# Sigmoid hyperparameter values.
alpha_sel = x_50
beta_sel = (x_80 - alpha_sel) / np.log((1-0.8)/0.8)

plt.figure()
plt.plot(x, inv_sigmoid(x, alpha_sel, beta_sel), label = 'alpha=' + 
         "{:.2f}".format(alpha_sel) + ', beta=' + "{:.2f}".format(beta_sel))
plt.scatter([x_80], [0.8], marker = '+', c = 'k', label = '80% of samples good-quality')
plt.scatter([x_50], [0.5], marker = 'x', c = 'k', label = '50% of samples good-quality')
plt.ylabel('P')
plt.xlabel('Human-evaluated sample quality rating')
plt.legend()
plt.ylim([0,1])
plt.xlim((x.min(), x.max()))
plt.title('Selected hyperparameters for P')
plt.show()


print('Choose alpha (midpoint) of ' + "{:.2f}".format(alpha_sel) + 
      ' and beta of ' + "{:.2f}".format(beta_sel) + ' for your optimization.')
###############################################################################
# NO CHANGES NEEDED
# Let's plot the effect of alpha and beta to the probability values.

beta_cand = np.arange(0.01, 1, 0.1)
alpha_cand = np.arange(0.01, 1, 0.1)

beta = beta_sel

for i in alpha_cand:
    
    c = np.array([1,1,1]) * i
    plt.plot(x, inv_sigmoid(x, i, beta), label = 'alpha=' + "{:.2f}".format(i),
             color = c)
    plt.scatter([i], [0.5], marker = 'x', color = c)
    
plt.ylabel('P')
plt.xlabel('Human evaluation')
plt.legend()
plt.title('Effect of Alpha (Beta = ' + "{:.2f}".format(beta) + ')')
plt.show()

alpha = alpha_sel

for i in beta_cand:
    
    c = np.array([1,1,1]) * i
    plt.plot(x, inv_sigmoid(x, alpha, i), label = 'beta=' + "{:.2f}".format(i),
             color = c)
    
plt.ylabel('P')
plt.xlabel('Human evaluation')
plt.legend()
plt.title('Effect of Beta (Alpha = ' + "{:.2f}".format(alpha) + ')')

plt.show()

#plot_P(model_human_retr, beta = 0.32, data_type = 'quality', midpoint = 0.44)
plot_P(model_human_retr, beta = beta_sel, data_type = 'quality', midpoint = alpha_sel)
