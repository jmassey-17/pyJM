# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:14:28 2023

@author: massey_j-adm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('mode.chained_assignment', None) #gets rid of errors that are unclear how to handle...
import seaborn as sns
from pyJM.BasicFunctions import timeOperation
import Stoner


def E(i,j,k): 
    """
    returns levi-civita value for i,j,k

    Parameters
    ----------
    i : int
    j : int
    k : int.

    Returns
    -------
    levi_civita value: int

    """
    return int((i-j)*(j-k)*(k-i)/2)

def calculateWindingNumber2(image, cx, cy, image_filled, window = 5):
    """
    will calculate the winding number in defined area + window
    for use with regionprops

    Parameters
    ----------
    image : np.array
        component of magnetization.
    cx : int
        centre in x direction of area.
    cy : int
        centre in x direction of area.
    image_filled : np.array
        result from regionprops.
    window : int, optional
        size of window around defined area. The default is 5.

    Returns
    -------
    windingNumber: float
        windingNumber of the area.

    """
    s = max(image_filled.shape)
    x1 = int(cx - int(s/2)) - window
    x2 = int(cx + int(s/2)) + window
    y1 = int(cy - int(s/2)) - window
    y2 = int(cy + int(s/2)) + window
    if y1 < 0: 
        y1,y2 = 0, s

    m = image[:,x1:x2, y1:y2]
    try:
        dmdx = np.gradient(m, axis = 1)
        dmdy = np.gradient(m, axis = 2)
        crossProduct = np.cross(dmdx, dmdy, axis = 0)
        dotProduct = np.sum(crossProduct*m, axis = 0)
        return (1/(4*np.pi))*np.sum(dotProduct)
    except: 
        return np.nan

class mumax:
    """
    Piggybacks Stoner.formats.simulations.OVFFile to generate magnetization array from mumax file
    that can be used to find and characterize Skyrmions
    """


    def __init__(self, filename, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards.

        Notes:
            This code can handle only the first segment in the data file.
        """
        self.filename = filename
        self.StonerObject = Stoner.formats.simulations.OVFFile(filename)
        self.metadata = self.StonerObject.metadata
        for m, i in zip(['mx','my', 'mz'],[3,4,5]):
            setattr(self, m, self.StonerObject.data.data[:,i].reshape((self.metadata['xnodes'],
                                                                       self.metadata['ynodes'], 
                                                                       self.metadata['znodes'])))
        self.m = np.array([self.mx,self.my,self.mz])
        

    def findAndCharacterizeSkyrmions(self): 
        """
        runs a regionprops for up and down and calcualtes skyrmion winding number for each area

        Returns
        -------
        None.

        """
        from skimage.measure import label, regionprops_table
        
        image = self.m[...,0] 
        positive = label(image[2] > 0)
        negative = label(image[2] < 0)
        types = ['positive', 'negative']
        
        for i, im in enumerate((positive, negative)): 
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'image_filled',
                                             'eccentricity',
                                             'axis_major_length',
                                             'axis_minor_length')))
            # Remove those that go off the screen
            offScreen_x = ((regions['bbox-0'] == 0) | (regions['bbox-2'] == im.shape[0]))
            offScreen_y = ((regions['bbox-1'] == 0) | (regions['bbox-3'] == im.shape[1]))
            
            onScreen = regions[(~offScreen_x & ~offScreen_y)]
            if len(onScreen) > 0: 
                onScreen['Q'] = onScreen.apply(lambda row: calculateWindingNumber2(image, row['centroid-0'], row['centroid-1'], row['image_filled'], 5), axis = 1)
                onScreen['type'] = types[i]
            try: 
                test = self.onScreen
            except: 
                test = 0
                
            if i == 0 or test == 0: 
                self.skyrmionAnalysis = onScreen
            else: 
                self.skyrmionAnalysis = pd.concat([self.skyrmionAnalysis, onScreen], ignore_index = True)


def returnFieldFromFilename(file): 
    """
    returns field from filename

    Parameters
    ----------
    filename : str
    
    Returns
    -------
    field: float

    """
    if file[0] == 'n': 
        return np.round(float(file[1:file.find('_')]), 3)
    elif file[0].isnumeric() or file[0] == '-': 
        return np.round(float(file[:file.find('_')]), 3)
    else: 
        return 0

    
