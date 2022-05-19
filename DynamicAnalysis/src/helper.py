#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:43:22 2021

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
from sympy import Float
from sympy.core.rules import Transform
#%%
def get_data_bo(example):
    
    X_train = np.linspace(-1,1).reshape(-1,1)
    y_true = 4*X_train**2 +  5*X_train
    if example ==1:
        y_train = 4*X_train**2 
    elif example ==2:
        y_train = 2*X_train + 3
        
    df_t_train = pd.DataFrame()
    df_X_train = pd.DataFrame()
    df_y_train = pd.DataFrame()
    df_y_true = pd.DataFrame()
    df_t_train['time'] = 0
    for idx in range(X_train.shape[1]):
        df_X_train[f'x{idx}'] = X_train[:,idx].flatten()
    df_y_train['y'] = y_train.flatten()
    df_y_true['true'] = y_true.flatten()
    return df_t_train,df_X_train,df_y_train,df_y_true
    
def print_eq(symexpr0_simple,output_columns):

    output_name = output_columns[-1]
    if 'error' in output_columns:
        x_name = output_columns[0]+'-'+output_columns[1]
    else:
         x_name = output_columns[0]
         
    symexpr0= str(symexpr0_simple).replace('x0', '('+x_name +')')

    equation = f"{output_name} = {symexpr0}"
    print(equation)
    
    return equation

def get_data_desired_sensed(problem,controller_type=None):
    df = pd.read_csv('../data/'+problem+'.csv')

    ix =0
    for idx,icol in enumerate(df.columns):
        if icol == 'time':
            df_t_train=pd.DataFrame()
            df_t_train['time'] = df[icol].values
        elif icol.endswith('_output'):
            if ix ==0:
                df_X_train=pd.DataFrame()
                df_X_train[icol] = df[icol].values
            else:
                df_X_train=pd.concat( [df_X_train,df[icol]],axis=1)
            ix = ix + 1
        elif icol.endswith('plant_command'):
            df_y_train=pd.DataFrame()
            df_y_train[icol]=df[icol].values
        else:
            print('please check the columns')
    
    if controller_type!=None:
        df_X_train_errer = pd.DataFrame()
        df_X_train_errer['desired_output'] = df_X_train['desired_output']
        df_X_train_errer['sensed_output'] = df_X_train['sensed_output']
        df_X_train_errer['error'] =  df_X_train['desired_output']-df_X_train['sensed_output']  
        df_X_train = df_X_train_errer
    
    return df_t_train,df_X_train,df_y_train

def get_data(problem):
    df = pd.read_csv('../data/'+problem+'_data.csv')

    ix =0
    for idx,icol in enumerate(df.columns):
        if icol == 'Time':
            t_train = df[icol].values.reshape(-1,1)
        elif icol.startswith('input'):
            if ix ==0:
                X_train = df[icol].values.reshape(-1,1)
            else:
                X_train=np.hstack((X_train,df[icol].values.reshape(-1,1)))
            ix = ix + 1
        elif icol.startswith('output'):
            y_train=df[icol].values.reshape(-1,1)
        else:
            print('please check the columns')
    
    df_t_train = pd.DataFrame()
    df_X_train = pd.DataFrame()
    df_y_train = pd.DataFrame()
    df_t_train['time'] = t_train.flatten()
    for idx in range(X_train.shape[1]):
        df_X_train[f'x{idx}'] = X_train[:,idx].flatten()
    df_y_train['y'] = y_train.flatten()
    
    return df_t_train,df_X_train,df_y_train


def standardized(x):
    x_mean=x.mean(0)
    x_std=x.std(0)
    x_n = (x - x_mean)/x_std
    
    return x_n, x_mean, x_std

def de_standardized(x,x_n):
    x_mean=x.mean(0)
    x_std=x.std(0)
    
    x_de = x_std* x_n + x_mean
    
    return x_de, x_mean, x_std

def output(symexpr,func,problem,df_t_train,df_X_train,df_y_train,theta=None,state=None,controller_type=None):
    
    output_columns=df_X_train.columns.values.tolist()+df_y_train.columns.values.tolist()
    t_train = df_t_train.values
    y_train = df_y_train.values
    if 'error' in df_X_train.columns:
        X_train=df_X_train['error'].values.reshape(-1,1)
    else:
        X_train=df_X_train.values
    
    
    if len(theta)!=0:
        y_pred = func(theta,t_train.flatten(),X_train.flatten(),state)
        
    else:
        y_pred = func(*X_train.T)
        
    plt.figure()

    if controller_type in ['PID','PI', 'PD']:
            plt.plot(t_train,y_train,'.',ms=10,color='red',label='data')
            plt.plot(t_train,y_pred,'o',ms=2,color='blue',label='equation')
            plt.xlabel('time')
    else:
            plt.plot(X_train[:,0],y_train,'.',ms=10,color='red',label='data')
            plt.plot(X_train[:,0],y_pred,'o',ms=2,color='blue',label='equation')
            plt.xlabel('x0')

        
    plt.ylabel('plant_command')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig('../output/'+problem+'_results.pdf')  
    
    mse = np.mean((y_train.flatten() - y_pred.flatten())**2)        
 
        
    print("MSE = {}".format(round(mse,6)))
    
    
    symexpr0=print_eq(symexpr,output_columns)
    with open('../output/'+problem+'_equation.txt', "w") as output:
        output.write(symexpr0)
        output.write(f'\nMSE = {mse}')
        
        
def check_sem(file_name = '../data/SemAnnotation-updated.csv'):
    import os
    
    file_exsist= os.path.isfile(file_name)

    if file_exsist:
        sem_annotation = pd.read_csv(file_name)
    
        if 'PI-Controller' in sem_annotation['AnnotationType'].values:
            controller_type = 'PI'
        elif 'PD-Controller' in sem_annotation['AnnotationType'].values:
            controller_type = 'PD'
        elif 'PID-Controller' in sem_annotation['AnnotationType'].values:
            controller_type = 'PID'
        else:
            print('\n could not find controller type\n')
            controller_type = None
    else:
        print('\n SemAnnotation file does not exsist\n')
        controller_type = None
    return controller_type