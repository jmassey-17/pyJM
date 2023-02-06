# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:28:53 2022

@author: massey_j
"""

import numpy as np

t1 = [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]]
t2 = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

def solution(m): 
    """Identify terminal and transient states"""
    m = np.array(m)
    terminal = []
    transient = []
    for i in range(m.shape[0]): 
        if np.sum(m[i]) > 0:
            transient.append(i)
        else:
            terminal.append(i)
    
    """Identify terminal and transient transitions"""
    transitions = []

    for i in range(m.shape[0]):
        for j in range(m.shape[1]): 
            if m[i,j] > 0:
                transitions.append([i,j])
    
    terminalTransitions = []
    transientTransitions = []
    
    for tran in transitions: 
        if tran[-1] in terminal:
            terminalTransitions.append(tran)
        else:
            transientTransitions.append(tran)
    
    """MODIFY this to account for transient to transition transitions"""
    """Identify transition chains"""
    transitionChains = []
    for ter in terminalTransitions:
        for tran in transientTransitions:
            if tran[-1] == ter[0]:
                transitionChains.append([tran, ter])
            else:
                transitionChains.append([ter])
                
    """Calculate probability of the chains"""
    numerator = {t: [] for t in terminal}
    denominator = {t: [] for t in terminal}
    
    for chain in transitionChains:
        p = 1
        d = 1
        for transition in chain:
            p = p*m[transition[0]][transition[1]]
            d = d*sum(m[transition[0]])
        numerator[transition[-1]].append(p)
        denominator[transition[-1]].append(d)  
    
    """Normalize the fractional answers"""
    "gather denominators"
    denoms = []
    for t in terminal:
        for i in range(len(denominator[t])):
            denoms.append(denominator[t][i])
    
    d = max(denoms)
    
    "normalize values where denom != d"
    for t in terminal:
        for i in range(len(denominator[t])):
            if denominator[t][i] != d: 
                numerator[t][i] = int(numerator[t][i]*(d/denominator[t][i]))
                denominator[t][i]= d
                
    """Add together probabilities"""
    f = [np.sum(numerator[t]) for t in terminal]
    f.append(d)
    return f, transitions
    
    