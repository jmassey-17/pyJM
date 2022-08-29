# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:39:00 2022

@author: massey_j
"""

import numpy as np
from pyJM.ThreeDimensional import Lamni, LamniMulti, LamniOutput
import matplotlib.pyplot as plt
import pyvista as pv
import os
import glob

heatScans = ['310', '335', '375']
coolScans = ['440', '330', '300']

homedir = r'C:\Data\FeRh\Reconstructions_All_20220425\wxmcd'

"""Do a proper plan of how to incorporate the GUI classes and the parameters file for previous measurements """

paramDict = {'300': {'H or C': 'C', 
                      'Rot': -20, 
                      'Box': [75, 215, 40, 215], 
                      'thresh': 0.3, 
                      'thetaoffset': 0},
              '310': {'H or C': 'H', 
                      'Rot': 20, 
                      'Box': [75, 230, 25, 210], 
                      'thresh': 0.3, 
                      'thetaoffset': 4},
              '330': {'H or C': 'C', 
                      'Rot': -20, 
                      'Box': [70, 215, 45, 215], 
                      'thresh': 0.08, 
                      'thetaoffset': 0},
              '335': {'H or C': 'H', 
                      'Rot': 20, 
                      'Box': [60, 220, 35, 210], 
                      'thresh': 0.2, 
                      'thetaoffset': 5},
              '375': {'H or C': 'H', 
                      'Rot': 23, 
                      'Box': [65, 220, 40, 220], 
                      'thresh': 0.1, 
                      'thetaoffset': 5},
              '440': {'H or C': 'C', 
                      'Rot': -28, 
                      'Box': [90, 235, 50, 230], 
                      'thresh': 0.03, 
                      'thetaoffset': 24},
              }     


homedir = r'C:\Data\FeRh\Reconstructions_All_20220425\wxmcd'
os.chdir(homedir)
files = glob.glob('*')
files = [file for file in files if len(file) < 7]
        


rec = Lamni.Lamni(files[-6], homedir, paramDict = paramDict)
rec.calcCurl()
rec.CalculateVorticity('magProcessed')
