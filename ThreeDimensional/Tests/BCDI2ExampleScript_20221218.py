# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:39:07 2022

@author: massey_j
"""

from pyJM.BasicFunctions import *
import pyJM.ThreeDimensional.BCDI2 as bcdi

homedir = r'C:\Data\3D AF\NaOsO3_MaxIV_202212\BCDI'
    
os.chdir(homedir)

recs = bcdi.CDIResult('267', homedir, 'NaOsO3_test_20221216_100_5_40_5_20_100_radius_50', 0.01)