# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:52:38 2023

@author: massey_j
"""

import os
import glob
import numpy as np


import scipy
import scipy.io
from skimage.transform import rotate
from skimage import feature, filters
from scipy.ndimage import binary_fill_holes, binary_dilation
import matplotlib.pyplot as plt


from pyJM.BasicFunctions import *
from pyJM.ThreeDimensional.GUIClasses import *      

homedir = r'C:\Data\3D AF\NaOsO3_MaxIV_202212\Tomography\Pin1'
os.chdir(homedir)
files = glob.glob('*.mat')

onfile = files[2]
print('Loading: {}'.format(onfile))
r = scipy.io.loadmat(glob.glob('*.mat')[-1])
on = r['tomogram_delta']


offfile = files[0]
print('Loading: {}'.format(offfile))
r = scipy.io.loadmat(glob.glob('*.mat')[-1])
off = r['tomogram_delta']
def threshTomo(tomo, thresh): 
    return abs(tomo) > thresh*np.amax(abs(tomo))

    
mask = threshTomo(on, 0.1)

arrays = [on, off]
for array in arrays: 
    array[~mask] = 0
    array[array < 0] = 0
    array = array[100:250, 125:260,:]

def plotnk(t2, n, k):
    fig, ax = plt.subplots(1,2)
    im = filters.gaussian(t2[n], 1)
    arr = im[...,k]
    ax[0].plot(arr)
    av = np.zeros_like(arr)
    av[:] = np.median(arr)
    av2 = np.zeros_like(arr)
    av2[:] = np.mean(arr)
    ax[0].plot(av)
    ax[0].plot(av2)
    ax[1].imshow(im)
    test = np.arange(im.shape[0])
    test[:] = k
    ax[1].plot(test, np.arange(im.shape[0]), 'r')
