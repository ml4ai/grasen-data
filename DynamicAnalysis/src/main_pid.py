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
problems = ['simple_PI_run1', 'simple_PI_run2']


controller_type = check_sem()
#%% read data from file 

df_t_train=[]
df_X_train=[]
df_y_train=[]
for idx in range(len(problems)):
    df_t_train_i,df_X_train_i,df_y_train_i= get_data_desired_sensed(problems[idx],controller_type) 
    df_t_train.append(df_t_train_i)
    df_X_train.append(df_X_train_i)
    df_y_train.append(df_y_train_i)
#%%
 

symexpr,func,theta,state=get_equation(df_t_train,df_X_train,df_y_train,controller_type)

#%%
symexpr,func,theta, state =symbolic(df_t_train,df_X_train,df_y_train,mop=True,controller_type=controller_type)
#%% print out the results
for idx in range(len(problems)):
    output(symexpr,func,problems[idx],df_t_train[idx],df_X_train[idx],df_y_train[idx],theta,state,controller_type=controller_type)
    
#%%

 