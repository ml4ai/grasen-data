#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:52:01 2020

@author: 212731466
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
import random
np.random.seed(seed=0)
random.seed(0)
import deap
import copy
import sympy
import matplotlib.image as mpimg
from deap import base, creator, tools, gp, algorithms
import operator
from scipy import integrate
import time
import warnings
warnings.filterwarnings("ignore")
from helper import de_standardized, standardized 
from sympy import Float
from sympy.core.rules import Transform
from helper import print_eq
#%%


def _evaluate_rmse_scale(individual,toolbox,X_train,y_train_all,mop):

            
    func =toolbox.lambdify(expr=individual)
    if isinstance(X_train, list):
        for idx in range(len(X_train)):
            Yp_i=func(*X_train[idx].T)
            if idx ==0:
                Yp=Yp_i
                y_train = y_train_all[0].flatten()
            else:
                Yp = np.hstack((Yp,Yp_i))
                y_train = np.hstack((y_train, y_train_all[idx].flatten()))
    else:
        Yp = func(*X_train.T)
        y_train = y_train_all.flatten()
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))

        try:
            if np.isfinite(Q).all():
                (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, y_train, rcond=None)
            else:
                residuals=np.array([])
        except:
            residuals=np.array([])
        if residuals.size > 0:
            if mop:
                return residuals[0] / len(y_train),    len(individual)
            else:
                return residuals[0] / len(y_train),    

    individual.a = 0
    individual.b = np.mean(y_train)
    if mop:
        return np.mean((y_train - individual.b) ** 2),  len(individual)
    else:
        return np.mean((y_train - individual.b) ** 2),
    


def _creater(X_train,y_train,pset,npop,parsimony_prob,max_depth,mop):
    if mop:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
    pset=pset,a=float, b=float)     
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4) 
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("lambdify", gp.compile, pset=pset)
    
    toolbox.register("evaluate", _evaluate_rmse_scale,toolbox=toolbox,X_train=X_train,y_train_all=y_train,mop=mop)
    if mop:
        toolbox.register("select", tools.selNSGA2, nd='standard')
    else:                    
        toolbox.register("select", tools.selDoubleTournament, parsimony_size=parsimony_prob, fitness_size=7, fitness_first=True)    
    
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=max_depth))    
    
    pop0 = toolbox.population(n=npop)
    
    if mop:
        hof = tools.ParetoFront()
    else:
        hof = tools.HallOfFame(10)
        
    stats = tools.Statistics(lambda ind: ind.fitness.values)    
    stats.register("MSE", np.min, axis=0) 
    
    return pop0, toolbox, stats, hof

def _evaluate_rmse(individual,toolbox,X_train,y_train):
    func = toolbox.lambdify(expr=individual)
    y_sr = func(X_train)

    residual = y_sr - y_train.flatten()

    rms = np.mean(residual ** 2)
    return rms,
    
def setting(X_train,y_train,t_train,npop,parsimony_prob,max_depth,mop,controller_type,name):
    if isinstance(X_train, list):
        no_var=X_train[0].shape[1]
    else:
        no_var=X_train.shape[1]
    pset = gp.PrimitiveSet("MAIN", no_var)
    if isinstance(t_train,list):
        t = t_train[0].flatten()
    else:
        t = t_train.flatten()
   
    def min_f(x,y):
        try:
            return  np.minimum(x,y)
        except ZeroDivisionError:
            return 0.0 * x
    
    def max_f(x,y):
        try:
            return  np.maximum(x,y)
        except ZeroDivisionError:
            return 0.0 * x
                
    def div_f(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return left *0.0
         
    def sqr_f(x):
        try:
            return np.sqrt(x)
        except ZeroDivisionError:
            return 0.0 * x
                
    def abs_f(x):
        try:
            return np.abs(x)
        except ZeroDivisionError:
            return 0.0 * x
    
    def grad(x):
        if isinstance(x, np.ndarray):
            xd=np.gradient(x.flatten(),t.flatten())
        else:
            xd = x * 0.0
        return xd
    
    def integral(x):
        if isinstance(x, np.ndarray):
            xint=integrate.cumtrapz(x.flatten(),t.flatten(),initial=0)
        else:
            xint = x * 0.0
        return xint        
    
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    
    if controller_type=='PID':
        pset.addPrimitive(grad, 1)
        pset.addPrimitive(integral, 1)
    
    if controller_type=='PD':
        pset.addPrimitive(grad, 1)
        
    if controller_type=='PI':
        pset.addPrimitive(integral, 1)

        
    if controller_type=='general':
        pset.addPrimitive(sqr_f, 1, name="sqrt")
        pset.addPrimitive(div_f, 2, name="div")
        pset.addPrimitive(abs_f, 1, name="abs")
        pset.addPrimitive(min_f, 2, name="min")
        pset.addPrimitive(max_f, 2, name="max")

        
    if name == None:
        pset.addEphemeralConstant('c0', lambda: random.uniform(-5,5))  
    else:
        pset.addEphemeralConstant(name, lambda: random.uniform(-5,5))  
    
    for ivar in range(no_var):
        pset.renameArguments(**{'ARG'+str(ivar):'x'+str(ivar)})
        
    pop0, toolbox, stats, hof = _creater(X_train,y_train,pset,npop,parsimony_prob,max_depth,mop)

    return pset,pop0, toolbox, stats, hof

def symbolic(df_t_train,df_X_train,df_y_train,file_name=None,mop=True,normalize=False,controller_type=None,name=None):
    
    if isinstance(df_X_train,list):
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
    
    else:
        output_columns=df_X_train.columns.values.tolist()+df_y_train.columns.values.tolist()
        t_train=df_t_train.values
        X_train=df_X_train.values      
        y_train=df_y_train.values              
        
    if file_name==None:
        df_setting = pd.read_csv('setting.csv')
    else:
        try:
            df_setting = pd.read_csv(file_name)
        except:
            print('please provide setting file')

    igen = df_setting['igen'][0]
    npop = df_setting['npop'][0]    
    seed = df_setting['seed'][0]
    cxpb = df_setting['cxpb'][0]
    mutpb = df_setting['mutpb'][0]
    parsimony_prob = df_setting['parsimony_prob'][0]
    max_depth = df_setting['max_depth'][0]
    
    
    st_time = time.time()
    
    if normalize:
        y_train_n,y_train_mean,y_train_std = standardized(y_train)
    else:
        y_train_n = y_train
        y_train_mean = 0.0
        y_train_std = 1.0
             
    pset, pop0, toolbox, stats, hof = setting(X_train,y_train_n,t_train,npop,parsimony_prob,max_depth,mop,controller_type,name)
    
    if mop:
        pop, logbook = algorithms.eaMuPlusLambda(pop0, toolbox, mu=10, 
                                     lambda_=npop, 
                                     cxpb=cxpb,
                                     mutpb=mutpb, 
                                     stats=stats, 
                                     ngen=igen, halloffame=hof,
                                     verbose=True)      
        fronts = tools.emo.sortLogNondominated(hof, len(hof))

    else:
        pop, logbook = algorithms.eaSimple(pop0, toolbox, cxpb, mutpb, igen,
        stats=stats, halloffame=hof, verbose=True)
        
    en_time = time.time()
    print("\n --- Elapsed time : %s seconds ---\n" % np.round((en_time - st_time),3))   
 
    if mop:
        ind =fronts[0][0]
    else:        
        ind = hof[0]
        
    symexpr0 = symbolic_print(ind)

 
    expr = y_train_std*(symexpr0*ind.a+ ind.b)+y_train_mean
    if isinstance(expr, list):
        symexpr0_simple=expr[0].xreplace(Transform(lambda x: x.round(3), lambda x: isinstance(x, Float)))
    else:
        symexpr0_simple=expr.xreplace(Transform(lambda x: x.round(3), lambda x: isinstance(x, Float)))
 
    #print("Best equation: u = {}".format(symexpr0_simple))
    
    symexpr0_text=print_eq(symexpr0_simple,output_columns)
    
    func = toolbox.lambdify(symexpr0_simple)
 
    if isinstance(X_train, list):
        for idx in range(len(X_train)):
            y_pred_i=func(*X_train[idx].T)
            if idx ==0:
                y_pred=y_pred_i
                y_train_all = y_train[0].flatten()
            else:
                y_pred = np.hstack((y_pred,y_pred_i))
                y_train_all = np.hstack((y_train_all,y_train[idx].flatten()))
                
        mse = np.mean((y_train_all - y_pred)**2)        
       
    else:         
        y_pred = func(*X_train.T)
        if isinstance(y_pred, np.ndarray):
            mse = np.mean((y_train - y_pred.reshape(-1,1))**2)
        else:
            mse = np.mean((y_train - y_pred)**2)
        
    print("MSE = {}".format(round(mse,6)))
    
    return symexpr0_simple,func,[],[]

 

def _convert_inverse_prim(prim, args):

    prim = copy.copy(prim)


    converter = {
        "sub": lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        "protectedDiv": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        "mul": lambda *args_: "Mul({},{})".format(*args_),
        "add": lambda *args_: "Add({},{})".format(*args_),
        "neg": lambda *args_: "Mul(-1,{})".format(*args_),
        "div": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
    }

    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)

def stringify_for_sympy(expression):
    string = ""
    stack = []
    for node in expression:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = _convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break   
            stack[-1][1].append(string)
    return string


def symbolic_print(expression, save_as= None):
 
    if isinstance(expression, str):
        symexpr = sympy.simplify(expression)
    else:
        stringify_symexpr = stringify_for_sympy(expression)
        symexpr = sympy.simplify(stringify_symexpr)

    if save_as is not None:
        sympy.init_printing()
        print(symexpr)
        sympy.preview(symexpr, viewer="file", outputTexFile=save_as + ".tex")
        sympy.preview(symexpr, viewer="file", filename=save_as + ".png")
        img = mpimg.imread(save_as + ".png")
        plt.figure()
        plt.axis("off")
        plt.imshow(img)
    return symexpr