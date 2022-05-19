#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:51:35 2022

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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import gridspec

#%%
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y,utility):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x0"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_ylabel('f(x)')
    axis.set_xlabel('x')
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_ylabel('Utility')
    acq.set_xlabel('x')
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    

def bo_setting(func,theta,x_data,y_data,kappa):
    
    def ss_function(x0):
        u=func(theta,x0)
        residual = y_data.flatten() - u.flatten()                
        mse = - np.mean(residual ** 2 )
        return mse
    
    
    optimizer = BayesianOptimization(
        f=ss_function,
        pbounds={'x0': (x_data.min(), x_data.max())},
        verbose=2,
        random_state=0,
    )
    
    utility_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0.0)
    
    next_point_to_probe = optimizer.suggest(utility_function)
    print("Next point to probe is:", next_point_to_probe)
    
    target = ss_function(**next_point_to_probe)
    print("Found the target value to be:", target)
    
    optimizer.register(
        params=next_point_to_probe,
        target=target,
    )
    
    return optimizer, utility_function,ss_function,x_data.min(),x_data.max()

def bo_sampling(func,theta,x_data,y_data,nsamples=50,PLOT=False,kappa=2.5):
    #%%
    optimizer, utility_function,ss_function,lower,upper = bo_setting(func,theta,x_data,y_data,kappa)
    n_points = 100

    x= np.linspace(lower,upper,n_points).reshape(-1,1)

    y = np.zeros((n_points,1))
    for idx in range(n_points):
        y[idx] = ss_function(x[idx])

    plt.plot(x,y)
    plt.title('objective function')
#%%
 
    for _ in range(nsamples):
        next_point = optimizer.suggest(utility_function)
        target = ss_function(**next_point)
        optimizer.register(params=next_point, target=target)
        utility = utility_function.utility(x, optimizer._gp, 0)
        if PLOT:
            plot_gp(optimizer, x, y,utility)
        print(target, next_point)
    
    # Finale state
    plot_gp(optimizer, x, y,utility)
    print(optimizer.max)

    import sampling
    X_BO = np.array(optimizer.max['params']['x0']).reshape(-1,1)
    y_BO = sampling.func_true(X_BO)
    
    return X_BO,y_BO
