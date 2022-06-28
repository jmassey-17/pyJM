# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:06:51 2022

@author: massey_j
"""


from pyJM.BasicFunctions import *
import pyJM.ThreeDimensional.BCDI2 as bcdi

"""1 - Load in of data 262"""
scansHeat = [912811, 912826, 912837, 912847, 912858, 912869, 
         912880, 912891, 912902, 912913, 912924, 912935]
scansCool = [912946, 912957, 912968, 912979, 912990, 
             913001, 913012, 913023, 913034]
tempDict = {'912811':300, #0
            '912826': 320,
            '912837': 340, #2
            '912847': 360,
            '912858': 380, #4
            '912869': 390,
            '912880': 395, #6
            '912891': 400,
            '912902': 405,
            '912913': 410,
            '912924': 415,
            '912935': 420, 
            '912946': 415, 
            '912957': 410, 
            '912968': 405, 
            '912979': 400, 
            '912990': 395, 
            '913001': 390, 
            '913012': 380,
            '913023': 360, 
            '913034': 340
           }

threshDict = {scan: 0.6 for scan in scansHeat[1:]}


homedir = r'C:\Data\3D AF\NaOsO BEamtime\Analysed_JM'
    
os.chdir(homedir)

recs = bcdi.CDIResults(scansHeat[1:], homedir, '262_300KStart_History_fromexec_100_5_40_5_20_100', threshDict)