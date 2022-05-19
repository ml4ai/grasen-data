#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:56:47 2021

@author: 212731466
"""
from util_multi import symbolic
from helper import get_data, standardized,output, check_sem,get_data_bo
from sem_annotation import get_equation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% read data from file 


df_t_train,df_X_train,df_y_train,df_y_true= get_data_bo(example=2) 

symexpr,func,theta, state =symbolic(df_t_train,df_X_train,df_y_train,mop=True,controller_type=None,name='c0')

#%% For now let's assume we know where is the constant


def func0(theta,X_train):
    # this function is the resulting from symexpr from SR
    return theta[0]*X_train+theta[1]

theta0=np.array([2.0, 3.0])
std=0.25
nsamples = 100
samples = np.random.uniform(theta0-theta0*std,theta0+theta0*std,(nsamples,len(theta0)))

y_samples=np.zeros((len(df_X_train),nsamples))
for isamples in range(nsamples):
    y_samples[:,isamples]= func0(samples[isamples],df_X_train.values).flatten()

#%%
plt.plot(df_X_train,df_y_train,'.',label='sample 0')
plt.plot(df_X_train,y_samples.mean(1),label='SR state 0')
plt.fill_between(df_X_train.values.flatten(), y_samples.mean(1)+y_samples.std(1),y_samples.mean(1)-y_samples.std(1),alpha = 0.5)
plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
error = np.mean((df_y_train.values.flatten()-y_samples.mean(1))**2).round(3)
print(f'MSE between the train and and current SR estimate={error}')
#%%
sample1=4*df_X_train.values**2 +  5*df_X_train.values
plt.plot(df_X_train.values,y_samples.mean(1),label='SR state 0')
plt.fill_between(df_X_train.values.flatten(), y_samples.mean(1)+y_samples.std(1),y_samples.mean(1)-y_samples.std(1),alpha = 0.5)
plt.plot(df_X_train.values,sample1,'+',label='sample 1')
plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
error = np.mean((sample1.flatten()-y_samples.mean(1))**2).round(3)
print(f'MSE between the train and and current SR estimate={error}')

#%%
from gaussian_process import Multifidelity_GP

y_high = sample1
y_low = y_samples.mean(1).reshape(-1,1)

model=Multifidelity_GP(df_X_train.values, y_low, df_X_train.values, y_high)

model.train()
#%%


y_gp_mean,y_gp_cov = model.predict(df_X_train.values)
y_gp_mean =y_gp_mean.flatten()

plt.plot(df_X_train.values.flatten(),y_gp_mean,label='MF')
plt.fill_between(df_X_train.values.flatten(),y_gp_mean+np.sqrt(y_gp_cov.diagonal()),y_gp_mean-np.sqrt(y_gp_cov.diagonal()),alpha = 0.5)

plt.plot(df_X_train.values,sample1,'+',label='sample 1')

plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')

error = np.mean((y_gp_mean-df_y_true.values.flatten())**2).round(3)
print(f'MSE between the new realization and SR+delta current estimate={error}')


 
#%% New data for new SR
y_delta = y_gp_mean-y_samples.mean(1)
 

df_y_delta=pd.DataFrame()
df_y_delta['delta'] = y_delta.flatten()
symexpr,func,theta, state =symbolic(df_t_train,df_X_train,df_y_delta,mop=True,controller_type=None,name='c1')


#%%
def func1(theta,X_train):
    # this function is the resulting from symexpr from SR
    return theta[0]*X_train**2  + theta[1]*X_train + theta[2]

theta1=np.array([3.986, 2.9895, -2.975])
std=0.25
nsamples = 100
samples = np.random.uniform(theta1-theta1*std,theta1+theta1*std,(nsamples,len(theta1)))

y_samples1=np.zeros((len(df_X_train),nsamples))
for isamples in range(nsamples):
    y_samples1[:,isamples]= func1(samples[isamples],df_X_train.values).flatten()
    
 
y_pred = func1(theta1,df_X_train.values)
plt.plot(df_X_train.values,y_pred,'+',label='SR state 1')
plt.plot(df_X_train.values,y_delta,'.',label='delta')
plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
error = np.mean((y_delta.flatten()-y_pred.flatten())**2).round(3)
print(f'MSE between the train and and current SR estimate={error}')
    
#%%
theta01=np.hstack([theta0, theta1])
def func0p1(theta,X_train):
    u0 = func0(theta[0:len(theta0)],X_train)
    u1 = func1(theta[len(theta0):],X_train)
    return u0+u1
    
std=0.25
nsamples = 100
samples = np.random.uniform(theta01-theta01*std,theta01+theta01*std,(nsamples,len(theta01)))

y_samples01=np.zeros((len(df_X_train),nsamples))
for isamples in range(nsamples):
    y_samples01[:,isamples]= func0p1(samples[isamples,:],df_X_train.values).flatten()


updated_mean = y_samples01.mean(1) 
updated_std =  y_samples01.std(1) 
plt.plot(df_X_train.values,updated_mean,label='SR state01')

plt.fill_between(df_X_train.values.flatten(),updated_mean+updated_std,updated_mean-updated_std,alpha = 0.5)
plt.plot(df_X_train.values,sample1,'+',label='sample 1')

plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')

error = np.mean((df_y_true.values.flatten()-updated_mean)**2).round(3)
print(f'MSE between the train and and current SR estimate={error}')
 
 