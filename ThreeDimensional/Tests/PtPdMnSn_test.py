# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:35:10 2022

@author: massey_j
"""

import os
import glob
from pyJM.ThreeDimensional.Lamni_20220527 import *
from pyJM.BasicFunctions import *


homedir = r'C:\Data\3D Skyrmion\PtPdMnSn\Filtered'
os.chdir(homedir)
files = glob.glob('*')

paramDict =  {'891' : {'H or C': 'H', 
                         'Rot': 0, 
                         'Box': [100, 380, 100, 380], #xx,yy
                         'thresh': 0.02,  
                         'thetaoffset': 0},
              '898' : {'H or C': 'H', 
                       'Rot': 0, 
                      'Box': [100, 380, 100, 380], #xx,yy
                                       'thresh': 0.08,  
                                       'thetaoffset': 0},
             }

#{'898' : {'H or C': 'H', 
#                      'Rot': 0, 
#                      'Box': [150, 450, 100, 350], #xx,yy
#                      'thresh': 0, 
#                      'thetaoffset': 0},
            
#paramDict = None
recs = Lamni(files[1], homedir, paramDict)
# recs.QuiverPlotSingle('x', 100, 10, 10, scale2 = 0.0001, pos = [-.9,1, 0.5, 0.5])
# recs.CalculateVorticity()
# vv = np.sqrt(np.sum(recs.vorticity**2, axis = 0))