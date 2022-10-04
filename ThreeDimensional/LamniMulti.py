# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:40:59 2022

@author: massey_j
"""
import os
import glob
import numpy as np

import scipy
import scipy.io
from skimage.transform import rotate
from skimage import feature
from scipy.ndimage import binary_fill_holes, binary_dilation


from pyJM.BasicFunctions import *


from pyJM.ThreeDimensional.Lamni import Lamni

class LamniMulti(Lamni): 
    """Class to load and view the results of magnetic laminography scans
    from the Donnelly matlab reconstruction code.
    """
    def __init__(self, homedir, paramDict):
        """Initial loading in procedure
        inputs: 
        - homedir: directory where the folders containing the reconstructions are found
        - paramDict: dictionary for each temperature with values for: 
            - 'H or C': heat or cool
            - 'Rot': angle through which to rotate the reconstruction so its straight
            - 'Box': box to crop the data too
            - 'thresh': Percentage threshold
            - 'thetaoffset': value which to roll the array through so all the angles line up
        
        IMPORTANT: 
            - Remove self.JM_FeRh_LamniSpecific when necessary
        """
        """Set up dictionaries for the data"""
        self.Params = paramDict
        self.recDict = {}
        self.projCalc = {}
        self.projMeas = {}
        self.thetaDict = {}
        
        """Load in the data for each of the data sets"""
        os.chdir(homedir)
        files = glob.glob('*')
        files.remove('AnalysisParams.csv')
        for file in files:
            if file[-1] != 'k': 
                super().__init__(file, homedir, paramDict, t = str(file[:3]))
        
        """ Calculate relevant quantities"""
        self.initializeMagneticArray()
        self.initializeMagneticDomains()
        self.zoomArrays()
                
    def JM_FeRh_LamniSpecific(self): 
        """Processes specific to JM experiment
            PLEASE REMOVE IF NEEDED"""
        for i in range(3):
            if i == 2:
                self.magProcessed['440'][i] = rotate(self.magProcessed['440'][i], 180)
            else: 
                self.magProcessed['440'][i] = -rotate(self.magProcessed['440'][i], 180)

        self.magMasks['440'] = rotate(self.magMasks['440'], 180) 
    
    def initializeMagneticArray(self): 
        """Initialize Magnetic arrays
        outputs: 
        - charge: charge values of the reconstruction
        - magProcessed: rotated, cropped and thresholded magnetic arrays
        - magDict: magnitude dictionary
        - magMasks: area's where threshold condition is met
        - chargeProcessed: charge values for masked areas
        - sampleOutline: outline of the sample for the volume calculation
        
        """
        
        self.charge = {}
        self.magProcessed = {}
        self.magDict = {}
        self.magMasks = {}
        self.chargeProcessed = {}
        self.sampleOutline = {}
        for t in list(self.Params.keys()):
            super().generateMagneticArray(self.Params[t]['Box'], self.Params[t]['thresh'], 200, outline = True, t = t)
        self.JM_FeRh_LamniSpecific()
        
    def zoomArrays(self): 
        """zoomed all magnetic ararys along the z direction so that 
        they're all the same height when making figures"""
        shape = np.zeros(shape = len(list(self.magProcessed.keys())))
        i = 0
        for t in list(self.magProcessed.keys()): 
            shape[i] = self.magProcessed['{}'.format(t)].shape[-1]
            i += 1
    
        maxPos = np.argmax(shape)

        self.zoomedDict = {}
        self.zoomedMasks = {}
        self.zoomedFinal = {}

        standard = self.magProcessed['{}'.format(list(self.magProcessed.keys())[maxPos])]
        maskStandard = self.magMasks['{}'.format(list(self.magProcessed.keys())[maxPos])]

        for t in list(self.magProcessed.keys()): 
            if t == list(self.magProcessed.keys())[maxPos]: 
                self.zoomedDict.update({'{}'.format(t): standard})
                self.zoomedMasks.update({'{}'.format(t): maskStandard})
            else: 
                self.zoomedDict.update({'{}'.format(t): zoom2(standard, self.magProcessed['{}'.format(t)])})
                self.zoomedMasks.update({'{}'.format(t): zoom2(maskStandard, self.magMasks['{}'.format(t)])})
            new = np.zeros_like(self.zoomedDict['{}'.format(t)])
            outline = np.where(self.zoomedMasks['{}'.format(t)] < -0.8)
            for i in range(3): 
                n = self.zoomedDict['{}'.format(t)][i]*(self.zoomedMasks['{}'.format(t)] > 0.8)
                n[outline] = np.nan
                new[i] = n
            self.zoomedFinal.update({'{}'.format(t): new})
    
    def volumeCalculation(self): 
        """Calculates volume from the magnetic/charge ratio"""
        self.volume = {}
        for t in list(self.magProcessed.keys()): 
            super().volumeCalc(t)
            
    def calculateCurl(self): 
        """Calculates mathematical curl"""
        self.curl = {}
        for t in list(self.magProcessed.keys()): 
            super().calcCurl(t)
            
    def initializeMagneticDomains(self):
        """Identifies the areas where domains point +/i in each of the three dircetions"""
        self.magDomains = {}
        for t in list(self.magProcessed.keys()): 
            super().magneticDomains(t)
    
    def saveParaviewAll(self, savePath): 
        """Saves paraview file for each of the temperatures"""
        for t in list(self.magProcessed.keys()): 
            super().saveParaview(self, savePath, t) 
    
    def calculateVorticity(self, attribute): 
        """Calculates magnetic vorticity for each temperature"""
        self.vorticity = {}
        for t in list(self.magProcessed.keys()): 
            super().CalculateVorticity(attribute, t)
            
    def filterAttribute(self, attribute, sigma): 
        self.filtered = {}
        for t in list(self.magProcessed.keys()): 
            super().filterAttribute(attribute, sigma, t)
    
    def calculateDirectionHistorgrams(self, binNo = 36): 
        self.direction = {}
        for t in list(self.magProcessed.keys()): 
            super().countPixelDirection(binNo, t)
        
