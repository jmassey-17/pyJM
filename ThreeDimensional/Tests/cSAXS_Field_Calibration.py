# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:44:58 2023

@author: massey_j
"""

import numpy as np
import matplotlib.pyplot as plt

def findFieldAt(distance, xnew, ynew):
    m = xnew == distance
    return f'Field at {distance} is {np.round(ynew[m][0], 1)} mT'

def fieldAtSample(distance, xnew, ynew, offset): 
    d = np.round(distance-offset, 2)
    m = xnew == d
    return f'Field at sample for {distance} is {np.round(ynew[m][0], 1)} mT'

x = np.array([-2, -1.5, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
offset = 0.25
y = np.array([30, 37, 47, 52.1, 57.9, 64.7, 72.8, 82.6, 94.4, 109, 127.5, 151.2, 182.5, 225, 283.8, 365.6, 472])

distances = np.array([-1.95, -1.1, -0.6, -0.3, 0, 0.2, 0.4, 0.5, 0.65, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.25, 1.3, 1.35])
fields = np.arange(30,210,10)
# what are the units of the dev.lmagnet?
# how does this convert to real distance?
# think the unit is mm, which correspond to 1.1 = 2600000

xnew = np.round(np.arange(x[0]-1, x[-1], 0.01), 2)
ynew = np.interp(xnew, x, y)
plt.plot(x,y, 'ro')
plt.plot(xnew, ynew, 'b-')

print(fieldAtSample(1.7, xnew, ynew, offset))