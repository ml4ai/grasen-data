#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:31:44 2022

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
#%%

def func_true(X):
    return np.minimum(0.5,np.maximum(-0.5,X))

def get_true(npoints=50):
    X_true = np.linspace(-1,1,npoints).reshape(-1,1)
    y_true = func_true(X_true)
    return X_true,y_true

def get_sample(xa):
    X_true,y_true=get_true()
    X_sample = X_true[X_true<=xa].reshape(-1,1)
    y_sample = y_true[X_true<=xa].reshape(-1,1)
    return X_sample,y_sample

def uq_propagation(func,theta0,X_sample,std=0.25, nsamples = 100):
    samples = np.random.uniform(theta0-theta0*std,theta0+theta0*std,(nsamples,len(theta0)))

    y_samples=np.zeros((len(X_sample),nsamples))
    for isamples in range(nsamples):
        y_samples[:,isamples]= func(samples[isamples],X_sample).flatten()
    
    y_mean=y_samples.mean(1)
    y_std=y_samples.std(1)
    return y_mean, y_std