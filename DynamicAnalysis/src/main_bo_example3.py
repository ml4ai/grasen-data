#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:56:47 2021

@author: 212731466
"""
from util_multi import symbolic
from helper import get_data, standardized,output, check_sem
from sem_annotation import get_equation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sampling 
import bo
#%% read data from file 

X_true, y_true = sampling.get_true()
plt.plot(X_true,y_true,'k-',label='actual')



X_sample0, y_sample0 =sampling.get_sample(1)

plt.plot(X_sample0,y_sample0,'o',label='sample0')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(0.25)
plt.legend()

#%%  
X_train = X_sample0
y_train = y_sample0

df_t_train0=pd.DataFrame()
df_X_train0=pd.DataFrame()
df_y_train0= pd.DataFrame()
 

df_t_train0['time']=0
df_X_train0['x'] = X_train.flatten()
df_y_train0['y'] = y_train.flatten()

symexpr,func,theta, state =symbolic(df_t_train0,df_X_train0,df_y_train0,mop=True,controller_type=None,name='c0')

#%% For now let's assume we know where is the constant
def func0(theta,x0):
    return -theta[0]*x0*(x0**2 - theta[1])
 

theta0=np.array([0.335,2.671])

# get a new sample
X_sample1, y_sample1 =sampling.get_sample(1)


y_mean0, y_std0= sampling.uq_propagation(func0,theta0,X_sample1)
#%%
plt.plot(X_true,y_true,'k-',label='actual')
plt.plot(X_sample1,y_sample1,'ro',label='sample1')
plt.plot(X_sample1,y_mean0,'b',label='SR state 0')
plt.fill_between(X_sample1.flatten(), y_mean0+y_std0,y_mean0-y_std0,color='b',alpha = 0.5)
plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
 
 
#%%
 
X_BO0,y_BO0=bo.bo_sampling(func0,theta0,X_sample1,y_sample1)

#%%

plt.plot(X_true,y_true,'k-',label='actual')

plt.plot(X_sample0,y_sample0,'o',label='sample0')

plt.plot(X_BO0,y_BO0,'s', label='BO sample')

plt.xlabel('x')
plt.ylabel('y')
plt.grid(0.25)
plt.legend()
#%%
X_train1 = np.vstack((X_sample0, X_BO0))
y_train1 = np.vstack((y_sample0, y_BO0))
 

df_t_train1=pd.DataFrame()
df_X_train1=pd.DataFrame()
df_y_train1= pd.DataFrame()
 

df_t_train1['time']=0
df_X_train1['x'] = X_train1.flatten()
df_y_train1['y'] = y_train1.flatten()

symexpr,func,theta, state =symbolic(df_t_train1,df_X_train1,df_y_train1,mop=True,controller_type='general',name='c1')
#%%

def func1(theta,x0):
    return theta[0]*np.maximum(np.minimum(x0, 0.521), -0.506) - theta[1]
    
theta1=np.array([ 0.98, 0.004])

y_mean1, y_std1= sampling.uq_propagation(func1,theta1,X_sample1)

#%%
plt.figure(figsize=(7,5))
plt.plot(X_true,y_true,'k-',label='actual')
plt.plot(X_sample0,y_sample0,'g.',markersize=10,label='sample0')
plt.plot(X_sample1,y_sample1,'r.',label='sample1')
plt.plot(X_sample1,y_mean1,'b',label='SR state 1')
plt.fill_between(X_sample1.flatten(), y_mean1+y_std1,y_mean1-y_std1,color='b',alpha = 0.15)
plt.plot(X_BO0,y_BO0,'s',color='tab:orange', label='BO sample')
plt.legend(loc='upper left')
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
