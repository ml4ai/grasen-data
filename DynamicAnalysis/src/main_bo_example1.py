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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bo_plot import plot_gp
import pandas as pd
#%% define the problem 

X_true = np.linspace(-5,5).reshape(-1,1)

def func_true(X):

    return X**3+ X**2

y_true = func_true(X_true)
plt.plot(X_true,y_true,'k-',label='actual')

X_train0 = X_true[X_true<=-4].reshape(-1,1)
y_train0 = y_true[X_true<=-4].reshape(-1,1)

plt.plot(X_train0,y_train0,'o',label='sample0')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(0.25)
plt.legend()

df_t_train0=pd.DataFrame()
df_X_train0=pd.DataFrame()
df_y_train0= pd.DataFrame()
 

df_t_train0['time']=0
df_X_train0['x'] = X_train0.flatten()
df_y_train0['y'] = y_train0.flatten()
#%%  
symexpr,func,theta, state =symbolic(df_t_train0,df_X_train0,df_y_train0,mop=True,controller_type=None,name='c0')

#%% For now let's assume we know where is the constant
X_sample1= X_true[X_true<=2].reshape(-1,1)
y_sample1 = y_true[X_true<=2].reshape(-1,1)
 
def func0(theta,x0):
    # this function is the results from SR symexpr
    return theta[0]*x0**3 + theta[1]*x0**2 + theta[2]*x0 + theta[3]
# and these are the original paramaters in symexpr
theta0=np.array([1.122, 2.688, 7.75, 11.833])
std=0.25
nsamples = 100
samples = np.random.uniform(theta0-theta0*std,theta0+theta0*std,(nsamples,len(theta0)))

y_samples0=np.zeros((len(X_sample1),nsamples))
for isamples in range(nsamples):
    y_samples0[:,isamples]= func0(samples[isamples],X_sample1).flatten()
#%%
plt.plot(X_true,y_true,'k-',label='actual')
plt.plot(X_sample1,y_sample1,'ro',label='sample1')
plt.plot(X_sample1,y_samples0.mean(1),'b',label='SR state 0')
plt.fill_between(X_sample1.flatten(), y_samples0.mean(1)+y_samples0.std(1),y_samples0.mean(1)-y_samples0.std(1),color='b',alpha = 0.5)
plt.legend()
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
 
 
#%%
X_train1 = X_sample1
y_train1 = y_sample1

    
lower = X_train1[0]
upper = X_train1[-1]
def ss_function(x0):
    u=func0(theta0,x0)
    residual = y_train1.flatten() - u.flatten()                
    mse = - np.mean(residual ** 2 )
    return mse


optimizer = BayesianOptimization(
    f=ss_function,
    pbounds={'x0': (lower, upper)},
    verbose=2,
    random_state=0,
)

utility_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

next_point_to_probe = optimizer.suggest(utility_function)
print("Next point to probe is:", next_point_to_probe)

target = ss_function(**next_point_to_probe)
print("Found the target value to be:", target)

optimizer.register(
    params=next_point_to_probe,
    target=target,
)
#%%
n_points = 100
x= np.linspace(lower,upper,n_points).reshape(-1,1)

y = np.zeros((n_points,1))
for idx in range(n_points):
    y[idx] = ss_function(x[idx])

plt.plot(x,y)
plt.title('objective function')
#%%
 
for _ in range(50):
    next_point = optimizer.suggest(utility_function)
    target = ss_function(**next_point)
    optimizer.register(params=next_point, target=target)
    utility = utility_function.utility(x, optimizer._gp, 0)
    plot_gp(optimizer, x, y,utility)
    print(target, next_point)


#%%
plot_gp(optimizer, x, y,utility)
print(optimizer.max)

#%% Now we add the new sample to the database and rerun SR
X_BO = np.array(optimizer.max['params']['x0']).reshape(-1,1)
y_BO = func_true(X_BO)

plt.plot(X_true,y_true,'k-',label='actual')


plt.plot(X_train0,y_train0,'o',label='sample0')

plt.plot(X_BO,y_BO,'s', label='BO sample')

plt.xlabel('x')
plt.ylabel('y')
plt.grid(0.25)
plt.legend()
#%%
X_train1 = np.vstack((X_train0, X_BO))
y_train1 = np.vstack((y_train0, y_BO))

df_t_train1=pd.DataFrame()
df_X_train1=pd.DataFrame()
df_y_train1= pd.DataFrame()
 

df_t_train1['time']=0
df_X_train1['x'] = X_train1.flatten()
df_y_train1['y'] = y_train1.flatten()

symexpr,func,theta, state =symbolic(df_t_train1,df_X_train1,df_y_train1,mop=True,controller_type=None,name='c1')
#%%

def func1(theta,x0):
    return theta[0]*x0**3 + theta[1]*x0**2 + theta[2]
    
theta0=np.array([0.999, 0.993, 0.038])
std=0.25
nsamples = 100
samples = np.random.uniform(theta0-theta0*std,theta0+theta0*std,(nsamples,len(theta0)))

y_samples0=np.zeros((len(X_sample1),nsamples))
for isamples in range(nsamples):
    y_samples0[:,isamples]= func1(samples[isamples],X_sample1).flatten()
#%%
plt.figure(figsize=(7,5))
plt.plot(X_true,y_true,'k-',label='actual')
plt.plot(X_train0,y_train0,'go',markersize=10,label='sample0')
plt.plot(X_sample1,y_sample1,'r.',label='sample1')
plt.plot(X_sample1,y_samples0.mean(1),'b',label='SR state 1')
plt.fill_between(X_sample1.flatten(), y_samples0.mean(1)+y_samples0.std(1),y_samples0.mean(1)-y_samples0.std(1),color='b',alpha = 0.15)
plt.plot(X_BO,y_BO,'s',color='tab:orange', label='BO sample')
plt.legend(loc='upper left')
plt.grid(0.25)
plt.xlabel('x')
plt.ylabel('y')
 