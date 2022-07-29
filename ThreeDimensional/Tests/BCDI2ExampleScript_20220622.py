# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:39:07 2022

@author: massey_j
"""

from pyJM.BasicFunctions import *
import pyJM.ThreeDimensional.BCDI2 as bcdi

scans073 = [912229, 912231, #300
            912264, 912266, #320
            912281, 912283, #340
            912306, 912308, #360
            #912323, 912325, #380
            #912421, 912423, #390
           # 912445, 912447, #395
            ]
   
"Temperature of each scan"
tempDict = {'912229': 300, 
            '912231': 300,
            '912264': 320,
            '912266': 320,
            '912281': 340,
            '912283': 340,
            '912306': 360,
            '912308': 360,
            '912323': 380,
            '912325': 380,
            '912421': 390, 
            '912423': 390, 
            '912445': 395,
            '912447': 395,
           }

"Threshold dictionary"
threshDict = {912229: 0.05, #0 0.18 073
            912231: 0.05, #0.18
            912264: 0.8, #4 073 hasnt worked
            912266: 0.01,
            912281: 0.01, #8 073
            912283: 0.4,
            912306: 0.05, #12 073
            912308: 0.165,
            #'912323': 0.15, #16 073
            #'912325': 0.05,
            #'912421': 0.2, 
            #'912423': 0.01, 
            #'912445': 0.1,
            #'912447': 0.85,
             }

homedir = r'C:\Data\3D AF\NaOsO BEamtime\Analysed_JM'
    
os.chdir(homedir)

recs = bcdi.CDIResults(scans073, homedir, '073_300KStart_History_fromexec_100_5_40_5_20_100', threshDict)