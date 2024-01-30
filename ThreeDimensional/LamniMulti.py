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
    """
    A class that load, trim and process multiple outputs of the Donnelly 3D xmcd 
    reconstruction algarithm.
 
    Attributes
    ----------
    homedir : str
        directory where files sits.
    paramDict : dict
        dictionary of parameters for loading process. Can be none
    
    """
    
    def __init__(self, homedir, searchCriteria, paramDict):
        """
        Initializes the Lamni  multiobject

        Parameters
        ----------
        homedir : str
            directory where file sits.
        paramDict : dict
            dictionary of parameters for loading process. Can be none

        Returns
        -------
        None.
        
        """
        
        self.recDict = {}
        self.thetaDict = {}
        self.projCalc = {}
        self.projMeas = {}
        self.projMeasCorrected = {}
        
        folders = os.listdir(homedir)
        for folder in folders: 
            file = [f for f in os.listdir(os.path.join(homedir, folder)) if f.find(searchCriteria) != -1][0]
            fileToLoad = os.path.join(homedir, folder, file)
            print(f'Loading {folder[:3]} K')
            super().__init__(fileToLoad, paramDict, t = str(folder[:3]))
        
        """ Calculate relevant quantities"""
        self.initializeMagneticArray()
        self.zoomArrays()
                 
    
    def initializeMagneticArray(self): 
        """
        Runs through lamni.generateMagneticArray for all files

        Returns
        -------
        None.

        """
        
        self.charge = {}
        self.magProcessed = {}
        self.magDict = {}
        self.magMasks = {}
        self.chargeProcessed = {}
        self.sampleOutline = {}
        for t in list(self.Params.keys()):
            super().generateMagneticArray(t = t)

        
    def zoomArrays(self): 
        """
        zoomed all magnetic ararys along the z direction so that 
        they're all the same height when making figures
        

        Returns
        -------
        None.

        """
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
        """
        Runs through volume calculation for all values of t

        Returns
        -------
        None.

        """
        self.volume = {}
        for t in list(self.magProcessed.keys()): 
            super().volumeCalc(t = t)
            
    def calculateCurl(self): 
        """
        Runs through curl calculation for all values of t

        Returns
        -------
        None.

        """
        self.curl = {}
        for t in list(self.magProcessed.keys()): 
            super().calcCurl(t = t)
    
    def saveParaviewAll(self, savePath): 
        """
        saves paraview file for all t

        Parameters
        ----------
        savePath : str
            path to save files to.

        Returns
        -------
        None.

        """
        for t in list(self.magProcessed.keys()): 
            super().saveParaview(savePath, t) 
    
    def calculateVorticity(self, attribute): 
        """
        calculates vorticity for all t

        Parameters
        ----------
        attribute : str
            attribute to use in calculation.

        Returns
        -------
        None.

        """
        self.vorticity = {}
        for t in list(self.magProcessed.keys()): 
            super().CalculateVorticity(attribute, t)
            
    def filterAttribute(self, attribute, sigma): 
        """
        applies gaussian filter of size sigma to attribute for all t

        Parameters
        ----------
        attribute : str
            attribute to filter.
        sigma : float
            width of gaussian filter.

        Returns
        -------
        None.

        """
        self.filtered = {}
        for t in list(self.magProcessed.keys()): 
            super().filterAttribute(attribute, sigma, t)
    
            
    def countDistribution(self): 
        """
        runs through count dictribution of pixels pointing in a given directipom 
        for all t

        Returns
        -------
        None.

        """
        self.distribution = {}
        for t in list(self.magProcessed.keys()): 
            super().countDistribution(t)
    
    def domainAnalysis2(self, thresh = 1): 
        """
        runs through lamni.domainAnalysis2 for all t

        Parameters
        ----------
        thresh : int, optional
            area threshold for domains. The default is 1.

        Returns
        -------
        None.

        """
        self.domains2 = {}
        self.domains2individual = {}
        for t in list(self.magProcessed.keys()): 
            super().domainAnalysis2(thresh, t)
        self.finalizeDomain2Analysis()
        
    def finalizeDomain2Analysis(self): 
        """
        cleans and processes results of the domainanalysis2

        Returns
        -------
        None.

        """
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
        """
        splits dataframe by scans 

        Parameters
        ----------
        attribute : str
            dataframe to use.
        scans : list
            list of scans to use.
        sortAttribute : str, optional
            attribute to use to sort the dataframe. The default is 'temp'.

        Returns
        -------
        None.

        """
        df = getattr(self, attribute)
        final = df[df[sortAttribute] == scans[0]]
        for t in scans[1:]:
           final = final.append(df[df[sortAttribute] == t])
        self.sortedDF = final
        self.sortedDFInfo = {'attr': attribute, 
                             'scans': scans, 
                             'sortedby': sortAttribute}
        
    def defineSizes(self): 
        """
        define pixel size in nm

        Returns
        -------
        None.

        """
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
        """
        creates final heat, cool dataframe from domainAnalysis2

        Parameters
        ----------
        attrs : str
            type of scan direction - either heat or cool.
        scans : list, optional
            list of scans in each attr. The default is [[310,335,375], [300, 330, 440]].
        labels1 : list, optional
            label for the output. The default is ['fm','af'].
        labels2 : list, optional
            labels the output either heat or cool. The default is {310: 'heat', 300: 'cool'}.

        Returns
        -------
        None.

        """
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
        """
        generates probability of finding domains in array at categories

        Parameters
        ----------
        array : str
            finals array name.
        categories : list
            ilst of categories to look at .

        Returns
        -------
        None.

        """
        out = [np.sum(self.finals[array][cat] > 0)/len(self.finals[array][cat]) for cat in categories]
        outerr = [np.sum(self.finals[array][cat] > 0)/len(self.finals[array][cat])*np.sqrt(1/np.sum(self.finals[array][cat] > 0) + 1/len(self.finals[array][cat])) for cat in categories]
        self.probs = [categories, out, outerr]
        
    def calcDistributions(self): 
        """
        loops through to calcualte the distributions for all t

        Returns
        -------
        None.

        """
        self.distributions = {}
        for t in list(self.magProcessed.keys()): 
            super().calcDistributions(t)
        print('Distributions calculated successfully')

