#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:35:14 2021

@author: 212731466
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(seed=0)
random.seed(0)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams.update({'figure.autolayout': True})
from scipy import integrate
import time
from scipy.optimize import minimize
#%%
def print_pid(res, state,output_columns):
    m_p = np.round(res.x[0],3) * state[0]
    m_i = np.round(res.x[1],3) * state[1]
    m_d = np.round(res.x[2],3) * state[2]       

    output_name = output_columns[-1]
    if 'error' in output_columns:
        x_name = output_columns[0]+'-'+output_columns[1]
    else:
         x_name = output_columns[0]
    symexpr0= str(m_p)+' * ' + '('+x_name +')'+ ' + ' + str(m_i) +' * integral('+ x_name + ')' + ' + ' + str(m_d) +' * grad('+ x_name + ')'
    control = 'PID'

    if state[0]==0:
        symexpr0= str(m_i) +' * integral('+ x_name + ')' + ' + ' + str(m_d) +' * grad('+ x_name + ')'
        control = 'ID'
    if state[1]==0:
        symexpr0= str(m_p)+' * ' + '('+x_name +')'+ ' + ' + str(m_d) +' * grad('+ x_name + ')'
        control = 'PD'
    if state[2]==0:
        symexpr0= str(m_p)+' * ' + '('+x_name +')'+ ' + ' + str(m_i) +' * integral('+ x_name + ')' 
        control = 'PI'
    if (state[1]==0 and state[2]==0):
        symexpr0= str(m_p)+' * ' + '('+x_name +')'
        control = 'P'

 
    print(f"{output_name} = {symexpr0}")
    
    return symexpr0
        
def model_PID(theta,t,x,state):

    u = state[0]*theta[0] * x + state[1]*theta[1]*integrate.cumtrapz(x,t,initial=0) + state[2]*theta[2] * np.gradient(x,t)
    return u

def ss_func(theta, *args):
    t_train=args[0]
    X_train=args[1]
    y_train=args[2]
    state = args[3]
    
    if isinstance(X_train, list):
        mse=np.zeros(len(X_train))
        for idx in range(len(X_train)):
            y_predict = model_PID(theta,t_train[idx].flatten(),X_train[idx].flatten(),state)
            mse[idx] = np.mean((y_train[idx].flatten()-y_predict.flatten())**2)
        
        return np.mean(mse)
        
    y_predict = model_PID(theta,t_train.flatten(),X_train.flatten(),state)

    mse = np.mean((y_train.flatten()-y_predict.flatten())**2)

    return mse

def get_equation(df_t_train,df_X_train,df_y_train,controller_type):
    
    # taking the columns from the first list
    output_columns=df_X_train[0].columns.values.tolist()+df_y_train[0].columns.values.tolist()
    
    t_train=[]
    X_train=[]
    y_train=[]
    for idx in range(len(df_X_train)):
        t_train_i = df_t_train[idx].values
        y_train_i = df_y_train[idx].values
        if 'error' in df_X_train[idx].columns:
            X_train_i=df_X_train[idx]['error'].values.reshape(-1,1)
        else:
            X_train_i=df_X_train[idx].values[:,0].reshape(-1,1)
        t_train.append(t_train_i)
        X_train.append(X_train_i)
        y_train.append(y_train_i)
    
    st_time = time.time()
    opt = {'maxiter' : 500,
               'gtol': 1e-12,
               'disp': False}

    bnds = ((-1000, 1000),(-1000, 1000),(-1000, 1000))
    theta0=np.array([1,1,1])

    if controller_type==None:
        return [], [], [], []        

    if controller_type!=None:
        
        if controller_type=='PI':
            state=(1,1,0)
        elif controller_type=='PD':
            state=(1,0,1)        
        elif controller_type=='P':
            state=(1,0,0)
        elif controller_type=='PID':
            state=(1,1,1) 
        else:
            state=(0,0,0)         
        if sum(state)==0:
            print('\n could not find controller type\n')
            return  
        
        res = minimize(ss_func, theta0,method='L-BFGS-B',options=opt,args=(t_train,X_train,y_train,state),bounds=bnds)
        if res.status==0:        
            symexpr0=print_pid(res, state,output_columns)
            theta = np.round(res.x,3)
            
            if isinstance(X_train, list):
                mse=np.zeros(len(X_train))
                for idx in range(len(X_train)):
                    y_predict = model_PID(theta,t_train[idx].flatten(),X_train[idx].flatten(),state)
                    mse[idx] = np.mean((y_train[idx].flatten()-y_predict.flatten())**2)
                mse=np.mean(mse)
            else:
                y_predict = model_PID(theta,t_train.flatten(),X_train.flatten(),state)        
                mse = np.mean((y_train.flatten()-y_predict.flatten())**2)
                
            print("MSE = {}".format(round(mse,6)))
                
        en_time = time.time()
        print("\n --- Elapsed time : %s seconds ---\n" % np.round((en_time - st_time),3))                   
                
    return symexpr0, model_PID, theta, state