# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:15:49 2023

@author: massey_j
"""
import os
import numpy as np
from pyJM.HomebrewMC.core import *

class simulationAnalyser: 
    def __init__(self, directory, fileType): 
        self.loadDirectory(directory, fileType)
        
    def loadDirectory(self, directory, fileType): 
        self.results = {}
        for folder in os.listdir(directory): 
            self.results.update({folder: np.load(os.path.join(directory, folder, file)) for file in os.listdir(os.path.join(directory, folder)) if file.find(f'{fileType}') != -1})
    
    def calculateOrderParameter(self): 
        self.OP = {key: calculateSpinOP(self.results[key], translations) for key in list(self.results.keys())}
        self.totalOP = {key: np.mean(self.OP[key]) for key in list(self.OP.keys())}
    