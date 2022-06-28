# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:50:25 2022

@author: massey_j
"""

import os
import glob
import numpy as np
from scipy.fftpack import ifftshift, fftshift, fft2

from pyJM.BasicFunctions import *

class CDIResult: 
    def __init__(self, scan, homedir): 
        #self.scan_no = scan
        print('Loading: {}'.format(scan))
        os.chdir(homedir)
        f = glob.glob('*{}*'.format(scan))
        os.chdir(f[0])
        return fftshift(np.load('rec_obj_{}.npy'.format(scan))), reduceArraySize(fftshift(np.load('rec_obj_{}.npy'.format(scan))), 0.1, 1)


class CDIResult2:
    def __init__(self, scan, homedir, folderSignature, threshValue): 
        self.__LoadReconstruction__(scan, homedir, folderSignature)
        self.__Threshold__(threshValue)
    
    def __LoadReconstruction__(self, scan, homedir, folderSignature):
        os.chdir(homedir)
        print('Loading: {}'.format(scan))
        file = os.path.join(homedir, '{}_{}'.format(scan, folderSignature))
        os.chdir(file)
        self.scan = scan
        self.rawData = fftshift(np.load('rec_obj_{}.npy'.format(scan)))
        here = np.where(np.nansum(abs(self.rawData), axis = (0,1)) != 0)[0]
        new = self.rawData[...,here]
        self.reducedData = new
        
    def __Threshold__(self, threshValue): 
        h = np.copy(self.reducedData, order = "C")
        here = abs(h) > np.nanmax(h)*thresh
        h[~here] = 0 
        self.thresholdedData = h
        
        
class CDIResults2(CDIResult): 
    def __init__(self, scans, homedir, tempDict, threshDict):
        self.scan_nos = scans
        rec_dict = {}
        raw_dict = {}
        for scan in scans:
            raw, processed = super().__init__(scan, homedir)
            rec_dict.update({'{}'.format(scan): processed})
            raw_dict.update({'{}'.format(scan): raw})
        self.rawData = raw_dict
        self.recData = rec_dict
        self.threshDict = threshDict
        self.tempDict = tempDict
        print('CDI Results Object Initialized')
        
    def ThresholdAndCompare(self): 
        new, mask = arrayThresh(self.recData, self.threshDict, self.scan_nos)
        pos = self.scan_nos[::2]
        neg = self.scan_nos[1::2]
        diffs = np.zeros(shape = len(pos))
        i = 0
        for p,n in zip(pos, neg): 
            ap = np.arctan2(new['{}'.format(p)].imag, new['{}'.format(p)].real)*mask['{}'.format(p)]/np.pi
            an = np.arctan2(new['{}'.format(n)].imag, new['{}'.format(n)].real)*mask['{}'.format(n)]/np.pi
            lp = len(np.where(np.mean(ap, axis = (0,1)) !=0)[0])
            ln = len(np.where(np.mean(an, axis = (0,1)) !=0)[0])
            diffs[i] = lp-ln
            i += 1
        if np.sum(diffs) == 0: 
            print('All arrays are the same size')
            self.processedData = new
            self.processedMask = mask
        else: 
            here = np.where(diffs[i] != 0)
            print('Scans for {} K need fixing'.format(self.tempDict['{}'.format(here[0])]))
            
    def ThresholdAndCompareAuto(self, variationDown = 0.01, variationUp = 0.01): 
        new, mask = arrayThresh(self.recData, self.threshDict, self.scan_nos)
        pos = self.scan_nos[::2]
        neg = self.scan_nos[1::2]
        diffs = np.zeros(shape = len(pos))
        i = 0
        for p,n in zip(pos, neg): 
            ap = np.arctan2(new['{}'.format(p)].imag, new['{}'.format(p)].real)*mask['{}'.format(p)]/np.pi
            an = np.arctan2(new['{}'.format(n)].imag, new['{}'.format(n)].real)*mask['{}'.format(n)]/np.pi
            lp = len(np.where(np.mean(ap, axis = (0,1)) !=0)[0])
            ln = len(np.where(np.mean(an, axis = (0,1)) !=0)[0])
            diffs[i] = lp-ln
            while diffs[i] != 0: 
                if diffs[i] > 0: 
                    self.threshDict['{}'.format(n)] -= variationDown
                    new, mask = arrayThresh(self.recData, self.threshDict, self.scan_nos)
                    ap = np.arctan2(new['{}'.format(p)].imag, new['{}'.format(p)].real)*mask['{}'.format(p)]/np.pi
                    an = np.arctan2(new['{}'.format(n)].imag, new['{}'.format(n)].real)*mask['{}'.format(n)]/np.pi
                    lp = len(np.where(np.mean(ap, axis = (0,1)) !=0)[0])
                    ln = len(np.where(np.mean(an, axis = (0,1)) !=0)[0])
                    diffs[i] = lp-ln
                elif diffs[i] < 0: 
                    self.threshDict['{}'.format(n)] += variationUp
                    new, mask = arrayThresh(self.recData, self.threshDict, self.scan_nos)
                    ap = np.arctan2(new['{}'.format(p)].imag, new['{}'.format(p)].real)*mask['{}'.format(p)]/np.pi
                    an = np.arctan2(new['{}'.format(n)].imag, new['{}'.format(n)].real)*mask['{}'.format(n)]/np.pi
                    lp = len(np.where(np.mean(ap, axis = (0,1)) !=0)[0])
                    ln = len(np.where(np.mean(an, axis = (0,1)) !=0)[0])
                    diffs[i] = lp-ln
            i += 1
        print('All arrays are the same size')
        self.processedData = new
        self.processedMask = mask