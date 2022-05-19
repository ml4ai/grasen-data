#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:56:47 2021

@author: 212731466
"""
from util_multi import symbolic
from helper import get_data, standardized,output, check_sem, get_data_desired_sensed
from sem_annotation import get_equation
import numpy as np
import matplotlib.pyplot as plt
#%% define the problem
 
problem = 'thrust_fraction'
problem = 'corrected_speed'
problem = 'saturation_block_lower'
problem = 'saturation_block_upper'

controller_type='general'
#%% read data from file 

df_t_train,df_X_train,df_y_train= get_data(problem) 

#%%
symexpr,func,theta, state =symbolic(df_t_train,df_X_train,df_y_train,mop=True,controller_type=controller_type)
#%% print out the results

output(symexpr,func,problem,df_t_train,df_X_train,df_y_train,theta,state,controller_type=controller_type)
    
#%%

 