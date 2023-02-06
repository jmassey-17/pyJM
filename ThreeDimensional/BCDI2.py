# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:10:55 2022

@author: massey_j
"""

import os
import glob
import numpy as np
from scipy.fftpack import ifftshift, fftshift, fft2

from pyJM.BasicFunctions import imageViewer, centreArray

# homedir = r'C:\Data\3D AF\NaOsO BEamtime\Analysed_JM'
# scan = 911811
class CDIResult:
    """Class for easy analysis of the results of pynx reconstructions"""
    
    def __init__(self, scan, homedir, folderSignature, threshValue, t = None): 
        if t != None: 
            self.__LoadReconstruction__(scan, homedir, folderSignature, t)
            self.__Threshold__(scan, threshValue, t)
        else: 
            self.__LoadReconstruction__(scan, homedir, folderSignature, t)
            self.__Threshold__(scan, threshValue, t)
            self.__Crop__()
            self.__PhaseUnwrap__(scan, t)
    
    def __LoadReconstruction__(self, scan, homedir, folderSignature, t = None):
        """Loads in the completed reconstruction saved as a .npy file
        inputs: 
            - scan: number of the scan
            - homedir: directory where the data can be found
            - folderSignature: end part of the folder name
            
        output: 
            - rawData: 
            - reducedData: raw data with the empty space either side of the reocnstructed object removed
        """
        os.chdir(homedir)
        print('Loading: {}'.format(scan))
        file = os.path.join(homedir, '{}_{}'.format(scan, folderSignature))
        os.chdir(file)
        
        if t != None: 
            "Loaded as a series of objects"
            self.rawData.update({scan: centreArray(fftshift(np.load('rec_obj_{}.npy'.format(scan))))})
        
        else: 
            "Loaded as a single object"
            self.scan = scan
            self.rawData = centreArray(fftshift(np.load('rec_obj_{}.npy'.format(scan))))
            
        
    def __Threshold__(self, scan, threshValue, t = None): 
        """Threshold for the intensity of the reduced data
        inputs: 
            - threshValue: fraction of intensity that is the minimum threshold
            
        output: 
            - Thresholded Data
        """
        if t != None: 
            h = np.copy(self.rawData[scan], order = "C")
            here = abs(h) > np.nanmax(abs(h))*threshValue
            h[~here] = 0 
            here2 = np.where(np.sum(np.angle(h), axis = (0,1)) != 0)[0]
            self.thresholdedData.update({scan: h[...,here2]})
        else: 
            h = np.copy(self.rawData, order = "C")
            here = abs(h) > np.nanmax(abs(h))*threshValue
            h[~here] = 0 
            here2 = np.where(np.sum(np.angle(h), axis = (0,1)) != 0)[0]
            self.thresholdedData = h[...,here2]
            
    def __Crop__(self): 
        p = np.where(abs(self.thresholdedData != 0))
        bounds = [min(p[1]),max(p[1]), min(p[0]),max(p[0]), min(p[2]),max(p[2])]
        size = [int(bounds[1] - bounds[0]), 
                int(bounds[3] - bounds[2]),
                int(bounds[5] - bounds[4])]
        new = np.zeros(shape = size, dtype = self.thresholdedData.dtype)
        new[:int(bounds[1]-bounds[0]),:int(bounds[3]-bounds[2]),:int(bounds[5]-bounds[4])] = self.thresholdedData[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]]
        self.cropped = centreArray(new)
        self.mask = abs(centreArray(new)) > 0
        
    def SliceViewer(self, keyWord, sliceNo, direction): 
        imageViewer(getattr(self, keyWord), sliceNo, direction)
        
    def __PhaseUnwrap__(self, scan, t = None): 
        from skimage.restoration import unwrap_phase
        if t != None: 
            self.unwrapped.update({scan: unwrap_phase(np.angle(self.cropped[scan]))*self.mask[scan]})
        else: 
            self.unwrapped = unwrap_phase(np.angle(self.cropped))*self.mask
        
class CDIResults(CDIResult): 
    def __init__(self, scans, homedir, folderSignature, threshDict):
        self.scans = scans
        self.rawData = {}
        self.thresholdedData = {}
        self.unwrapped = {}
        for scan in scans: 
            super().__init__(scan, homedir, folderSignature, threshDict[scan], t = True)
        
        self.__Crop__()
        for scan in scans: 
            super().__PhaseUnwrap__(scan, t = True)
        
    
    def __Crop__(self): 
        bounds = np.zeros(shape = (len(self.scans), 6), dtype = int)
        i = 0
        for scan in self.scans: 
            p = np.where(abs(self.thresholdedData[scan] != 0))
            bounds[i,:] = min(p[1]),max(p[1]), min(p[0]),max(p[0]), min(p[2]),max(p[2])
            i += 1
        size = [int(max(bounds[:, 1]) - min(bounds[:, 0])),
                int(max(bounds[:, 3]) - min(bounds[:, 2])),
                int(max(bounds[:, 5]) - min(bounds[:, 4]))]
        print(size, bounds)
        i = 0
        cropped = {}
        mask = {}
        for scan in self.scans: 
            new = np.zeros(shape = size, dtype = self.thresholdedData[scan].dtype)
            new[:int(bounds[i,1]-bounds[i,0]),:int(bounds[i,3]-bounds[i,2]),:int(bounds[i,5]-bounds[i,4])] = self.thresholdedData[scan][bounds[i,0]:bounds[i,1], bounds[i,2]:bounds[i,3], bounds[i,4]:bounds[i,5]]
            cropped.update({scan: new})
            mask.update({scan: abs(new)  > 0})
            i += 1
        self.cropped = cropped
        self.mask = mask
        
            
            
    def SliceViewer(self, keyWord, scan, sliceNo, direction): 
        imageViewer(getattr(self, keyWord)[scan], sliceNo, direction)
        
    def plotAllzSlices(self, keyWord, scan): 
        import matplotlib.pyplot as plt
        n = getattr(self, keyWord)[scan]
        if n.shape[2] % 2 == 0: 
            num = 2
        elif n.shape[2] % 3 == 0: 
            num = 3
        else: num = n.shape[2]
        fig,ax = plt.subplots(n.shape[2] // num, int(n.shape[2]/(n.shape[2] // num)), constrained_layout=True)
        fig.suptitle('All slices for {}'.format(scan))
        i, j = 0,0
        for k in range(n.shape[2]): 
            bar = ax[i,j].imshow(n[...,k]) #, vmin = -1, vmax = 1)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title('Slice {}'.format(k))
            j += 1
            if j == int(n.shape[2]/(n.shape[2] // num)):
                j = 0
                i += 1
        #cax = plt.axes([0.85, 0.1, 0.05, 0.8])
        #plt.colorbar(bar, cax = cax)
        plt.tight_layout()
        

                
            
        
        
        
    
    