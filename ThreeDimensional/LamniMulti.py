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
            shape[i] = self.magProcessed[f'{t}'].shape[-1]
            i += 1
    
        maxPos = np.argmax(shape)

        self.zoomedDict = {}
        self.zoomedMasks = {}
        self.zoomedFinal = {}
        self.zoomedMag = {}

        standard = self.magProcessed[f'{list(self.magProcessed.keys())[maxPos]}']
        maskStandard = self.magMasks[f'{list(self.magProcessed.keys())[maxPos]}']
        magStandard = self.magDict[f'{list(self.magProcessed.keys())[maxPos]}']

        for t in list(self.magProcessed.keys()): 
            if t == list(self.magProcessed.keys())[maxPos]: 
                self.zoomedDict.update({f'{t}': standard})
                self.zoomedMasks.update({f'{t}': maskStandard})
                self.zoomedMag.update({f'{t}': magStandard})
            else: 
                self.zoomedDict.update({f'{t}': zoom2(standard, self.magProcessed[f'{t}'])})
                self.zoomedMasks.update({f'{t}': zoom2(maskStandard, self.magMasks[f'{t}'])})
                self.zoomedMag.update({f'{t}': zoom2(maskStandard, self.magDict[f'{t}'])})
            new = np.zeros_like(self.zoomedDict[f'{t}'])
            outline = np.where(self.zoomedMasks[f'{t}'] < -0.8)
            for i in range(3): 
                n = self.zoomedDict['{}'.format(t)][i]*(self.zoomedMasks['{}'.format(t)] > 0.8)
                n[outline] = np.nan
                new[i] = n
            self.zoomedFinal.update({'{}'.format(t): new})
    
    def volumeCalculation(self): 
        """Calculates volume from the magnetic/charge ratio"""
        self.volume = {}
        for t in list(self.magProcessed.keys()): 
            super().volumeCalc(t = t)
            
    def calculateCurl(self): 
        """Calculates mathematical curl"""
        self.curl = {}
        for t in list(self.magProcessed.keys()): 
            super().calcCurl(t = t)
            
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
            
    def countDistribution(self): 
        self.distribution = {}
        for t in list(self.magProcessed.keys()): 
            super().countDistribution(t)
    
    def domainAnalysis2(self, thresh = 10): 
        self.domains2 = {}
        self.domains2individual = {}
        for t in list(self.magProcessed.keys()): 
            super().domainAnalysis2(thresh, t)
        self.finalizeDomain2Analysis()
        
    def finalizeDomain2Analysis(self): 
        import pandas as pd
        temp_keys = list(self.domains2individual.keys())
        final_fm_ind = pd.DataFrame(self.domains2individual[temp_keys[0]]['fm'])
        final_af_ind = pd.DataFrame(self.domains2individual[temp_keys[0]]['af'])
        final_fm_all = pd.DataFrame(self.domains2[temp_keys[0]]['fm'])
        final_af_all = pd.DataFrame(self.domains2[temp_keys[0]]['af'])
        for t in temp_keys[1:]:
            final_fm_ind = final_fm_ind.append(self.domains2individual[t]['fm'])
            final_af_ind = final_af_ind.append(self.domains2individual[t]['af'])
            final_fm_all = final_fm_all.append(self.domains2[t]['fm'])
            final_af_all = final_af_all.append(self.domains2[t]['af'])
        final_fm_ind['temp'] = final_fm_ind['temp'].astype(np.float)
        final_af_ind['temp'] = final_af_ind['temp'].astype(np.float)
        final_fm_all['temp'] = final_fm_all['temp'].astype(np.float)
        final_af_all['temp'] = final_af_all['temp'].astype(np.float)
    

        self.finalIndividualFM = final_fm_ind
        self.finalIndividualAF = final_af_ind
        self.finalFM = final_fm_all
        self.finalAF = final_af_all
        
    def generateHeatCoolDataframe(self, attribute, scans, sortAttribute = 'temp'): 
        df = getattr(self, attribute)
        final = df[df[sortAttribute] == scans[0]]
        for t in scans[1:]:
           final = final.append(df[df[sortAttribute] == t])
        self.sortedDF = final
        self.sortedDFInfo = {'attr': attribute, 
                             'scans': scans, 
                             'sortedby': sortAttribute}
        
    def defineSizes(self): 
        sizes = {}
        for t in list(self.magProcessed.keys()): 
            d = self.sampleOutline[t] 
            x = len(np.nonzero(d[100,:,0])[0])
            y = len(np.nonzero(d[:,100,0])[0])
            z = d.shape[2]
            vol = np.sum(d)
            sizes.update({t: [6700/x, 8000/y, 145/z, vol]})

        self.sizes = sizes
        
    def generateFinalDataframe(self, attrs, scans = [[310,335,375], [300, 330, 440]], 
                               labels1 = ['fm','af'],
                               labels2 = {310: 'heat', 300: 'cool'}): 
        lfin = {}
        for a,l in zip(attrs, labels1): 
            lfin.update({a: l})
        
        finals = {}

        for a in attrs:
            for s in scans: 
                 self.generateHeatCoolDataframe(a, s)
                 finals.update({'{}_{}'.format(lfin[a], labels2[s[0]]): self.sortedDF})
        
        self.defineSizes()
        
        for key in list(finals.keys()):
            df = finals[key]
            df['xnew'] = df.apply(lambda row: abs(row['bbox-3']-row['bbox-0'])*self.sizes['{}'.format(int(row['temp']))][1], axis = 1)
            df['ynew'] = df.apply(lambda row: abs(row['bbox-4']-row['bbox-1'])*self.sizes['{}'.format(int(row['temp']))][0], axis = 1)
            df['znew'] = df.apply(lambda row: abs(row['bbox-5']-row['bbox-2'])*self.sizes['{}'.format(int(row['temp']))][2], axis = 1)
            df['volnew'] = df.apply(lambda row: row['area']/self.sizes['{}'.format(int(row['temp']))][3], axis = 1)
            df['both'] = df.apply(lambda row: (row['top'] == True)*(row['bottom'] == True), axis = 1)
            df['neither'] = df.apply(lambda row: (row['top'] == False)*(row['bottom'] == False), axis = 1)
            df['either'] = df.apply(lambda row: 1-row['neither'], axis = 1)
        
        self.finals = finals
        
    def generateProbability(self, array, categories): 
        out = [np.sum(self.finals[array][cat] > 0)/len(self.finals[array][cat]) for cat in categories]
        outerr = [np.sum(self.finals[array][cat] > 0)/len(self.finals[array][cat])*np.sqrt(1/np.sum(self.finals[array][cat] > 0) + 1/len(self.finals[array][cat])) for cat in categories]
        self.probs = [categories, out, outerr]
        
    def calcDistributions(self): 
        self.distributions = {}
        for t in list(self.magProcessed.keys()): 
            super().calcDistributions(t)
        print('Distributions calculated successfully')
    
    def calcAsym(self): 
        self.asym = {}
        for t in list(self.magProcessed.keys()): 
            super().calcAsymmetries(t)
        print('Asymmetries calculated successfully')
        
    def saveAsym(self, savePath, fileName = None, key = 'fm'): 
        for t in list(self.magProcessed.keys()): 
            super().saveAsym(savePath, fileName, key, t)
        print(f'Asymmetries saved in {savePath}')
        
    def saveDistribution(self, savePath, fileName = None, key = 'fm'): 
        for t in list(self.magProcessed.keys()): 
            super().saveDistribution(savePath, fileName, key, t)
        print(f'Distributions saved in {savePath}')