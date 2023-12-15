# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:06:35 2023

@author: massey_j
"""


import numpy as np
from scipy.optimize import minimize
import threading
from time import perf_counter
from joblib import Parallel, delayed

"""Strain tensor stuff"""

def func(u, reflections, phaseArrays, i,j,k): 
    s = 0
    for refl in list(phaseArrays.keys()): 
        s += (np.dot(reflections[refl], u) - phaseArrays[refl][i,j,k])**2
    return s

def calculateGrad(array): 
    return np.array([np.gradient(array, axis = k) for k in (0,1,2)])

def minimize_function(func, uGuess, result, *args):
    result[:, args[-3], args[-2], args[-1]] = minimize(func, uGuess, args=(args)).x

def minimize_function_parallel(func, uGuess, result, bnds, reflections, arrays, i,j,k):
    return (i,j,k, minimize(func, uGuess, bounds = bnds, args=(reflections, arrays, i,j,k)).x)


reflections = {'222': np.array([2,2,2]),
               '002': np.array([0,0,2]),
               '121': np.array([1,2,1]), 
               '101': np.array([1,0,1])}

np.random.seed = 1
xsize, ysize, zsize = 10, 10,10


arrays = {refl: np.random.random(size = (xsize,ysize,zsize))*2*np.pi - np.pi for refl in list(reflections.keys())}

phasegradient = {refl: calculateGrad(arrays[refl]) for refl in list(arrays.keys())}

uGuess = np.array([1,0,1])

bnds = ((-1, 1), (-1, 1), (-1,1))

"""This works"""
"""Thread for all at once and test performancee"""
result = np.zeros(shape =(3,xsize,ysize,zsize))

# start = perf_counter()
# for i in range(result.shape[1]): 
#     for j in range(result.shape[2]):
#         for k in range(result.shape[3]):
#             result[:,i,j,k] = minimize(func, uGuess, bounds = bnds, args=(reflections, arrays, i,j,k)).x
# stop = perf_counter()
# time = stop-start
size = result.shape[1]*result.shape[2]*result.shape[3]
# print(f'Array of size {size} processed in {time} seconds. Average speed = {time/size} seconds per point')

# s = perf_counter()
# result2 = np.zeros(shape = (3,xsize,ysize,zsize))
# threads = []
# for i in range(result2.shape[1]): 
#     for j in range(result2.shape[2]):
#         for k in range(result2.shape[3]):
#               t = threading.Thread(target=minimize_function, args=(func,  uGuess, result2, reflections, arrays, i,j,k,)) 
#               t.start()
#               # add append code here
#               threads.append(t)
#             # add join code here
#         for t in threads:
#           t.join()

# time = perf_counter() - s
# print(f"Threading: {time} seconds. Average speed = {time/size} seconds per point.")

"parallel test"
result3 = np.zeros(shape = (3,xsize,ysize,zsize))
num_cores = 4
start = perf_counter()
parallelResult = Parallel(n_jobs=num_cores)(delayed(minimize_function_parallel)(func, uGuess, result3, bnds, reflections, arrays, i,j,k) for i in range(result3.shape[1]) for j in range(result3.shape[2]) for k in range(result3.shape[3]))
time = perf_counter()-start

start = perf_counter()
for p in parallelResult: 
    result3[:,p[0],p[1], p[2]]  = p[-1]
rebuild = perf_counter() - start
print(f'Parallel: {time + rebuild} seconds. Average speed = {(time+rebuild)/size} seconds per point on {num_cores} cores.')

# class ThreadMinimizer(): 
#     def __init__(self, result, reflections, arrays): 
#         self.result = result
#         self.uGuess = np.array([1,0,1])
#         s = perf_counter()
#         print('Initiating sequence')
#         for i in range(self.result.shape[1]): 
#             for j in range(self.result.shape[2]):
#                 for k in range(self.result.shape[3]):
#                       t = threading.Thread(target=self.task, args=(self.func, self.uGuess, reflections, arrays, i,j,k,)) 
#                       t.start()
#                       # add append code here
#                       threads.append(t)
#                     # add join code here
#                 for t in threads:
#                   t.join()
#         elapsed = perf_counter() - s
#         print("Threading Elapsed Time: " + str(elapsed) + " seconds")
    
#     def task(self, *args):
#         var = minimize(self.func, self.uGuess, args=(args)).x
#         self.result[:, args[-3],args[-2], args[-1]] = var
#         print(self.result[:, args[-3],args[-2], args[-1]], print(var))
        
#     def func(u, reflections, arrays, i,j,k): 
#         s = 0
#         for refl in list(arrays.keys()): 
#             s += (np.dot(reflections[refl], u) - arrays[refl][i,j,k])**2
#         return s




