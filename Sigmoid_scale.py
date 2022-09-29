#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 01:34:19 2022

@author: armi
"""

plt.plot(np.arange(-1,1,0.1), 1/(1+np.exp((np.arange(-1,1,0.1))/0.1)))
plt.xlabel('Human evaluation')
plt.ylabel('Probability of a good sample')
plt.legend([r'$\beta$=0.1'])