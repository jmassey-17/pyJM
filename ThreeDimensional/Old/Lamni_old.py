# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:50:25 2022

@author: massey_j
"""
import os
import glob
import numpy as np
import csv
import threading

import scipy
import scipy.io
from skimage.transform import rotate
from skimage import feature
from scipy.ndimage import binary_fill_holes, binary_dilation
import matplotlib.pyplot as plt


from pyJM.BasicFunctions import *
from pyJM.ThreeDimensional.GUIClasses import *

"""Do a proper plan of how to incorporate the GUI classes and the parameters file for previous measurements """

# paramDict = {'300': {'H or C': 'C', 
#                      'Rot': -20, 
#                      'Box': [75, 215, 40, 215], 
#                      'thresh': 0.3, 
#                      'thetaoffset': 0},
#              '310': {'H or C': 'H', 
#                      'Rot': 20, 
#                      'Box': [75, 230, 25, 210], 
#                      'thresh': 0.3, 
#                      'thetaoffset': 4},
#              '330': {'H or C': 'C', 
#                      'Rot': -20, 
#                      'Box': [70, 215, 45, 215], 
#                      'thresh': 0.08, 
#                      'thetaoffset': 0},
#              '335': {'H or C': 'H', 
#                      'Rot': 20, 
#                      'Box': [60, 220, 35, 210], 
#                      'thresh': 0.2, 
#                      'thetaoffset': 5},
#              '375': {'H or C': 'H', 
#                      'Rot': 23, 
#                      'Box': [65, 220, 40, 220], 
#                      'thresh': 0.1, 
#                      'thetaoffset': 5},
#              '440': {'H or C': 'C', 
#                      'Rot': -28, 
#                      'Box': [90, 235, 50, 230], 
#                      'thresh': 0.3, 
#                      'thetaoffset': 24},
#              }           
      
class Lamni(): 
    
    """Class to load and view the results of magnetic laminography scans
    from the Donnelly matlab reconstruction code.
    - will work as a single or as part of the LamniMulti
    """
    def __init__(self, file, homedir, paramDict, t=None):
        
        """Initial loading in procedure
        inputs:
            
        - homedir: directory where the folders containing the reconstructions are found
        - paramDict: dictionary for each temperature with values for: 
            - 'H or C': heat or cool
            - 'Rot': angle through which to rotate the reconstruction so its straight
            - 'Box': box to crop the data too
            - 'thresh': Percentage threshold
            - 'thetaoffset': value which to roll the array through so all the angles line up
        - t determines if the object is a standalone reconstruction or as a group
        will be auto-assigned depending on whether its run as lamni or lamnimulti
        
        """
        """Load in the *mat file from the reconstructions"""
        os.chdir(homedir)
        print('Loading: {}'.format(file))
        os.chdir(file)
        r = scipy.io.loadmat(glob.glob('*.mat')[-1])
        
        """Crop it so only the non-masked elements in the mag are taken forward"""
        temp = r['mx_out']
        here = np.where(np.mean(temp, axis = (0,1)) != 0 )
        rec_charge = r['abs_out'][...,here[0]]
        rec_mag = np.array([r['mx_out'][...,here[0]], 
                            r['mz_out'][...,here[0]], 
                            r['my_out'][...,here[0]]])

        self.t = str(file[:3])
        self.rawCharge = rec_charge 
        self.rawMag = rec_mag
        
        if paramDict == None: 
            """Initialize the param dict for first time use"""
            paramDict = {'H or C': 'C', 
                         'Rot': 0, 
                         'Box': 0, 
                         'thresh': 0, 
                         'thetaoffset': 0}
            """Get the rotation"""
            happy = 'n'
            while happy == 'n': 
                rottest = int(input("Enter test value of the rotation in degrees: "))
                fig, ax = plt.subplots(1,2, num = 1)
                ax[0].imshow(abs(self.rawMag[0,...,0]))
                ax[1].imshow(scipy.ndimage.rotate(abs(self.rawMag[0,...,0]), rottest, reshape = False))
                fig.canvas.draw()
                plt.pause(0.05)
                happy = input('Happy? y or n: ')
                plt.close(1)
            paramDict['Rot'] = rottest 
            """get the box"""
            happy = 'n'
            while happy == 'n': 
                boxTest = [0,0,0,0]
                boxTest[0] = int(input("Enter the xlow estimate: "))
                boxTest[1] = int(input("Enter the xhigh estimate: "))
                boxTest[2] = int(input("Enter the ylow estimate: "))
                boxTest[3] = int(input("Enter the  estimate: "))
                fig, ax = plt.subplots(1,2, num = 1)
                ax[0].imshow(abs(self.rawMag[0,...,0]))
                ax[1].imshow(abs(self.rawMag[0,boxTest[2]:boxTest[3], boxTest[0]:boxTest[1], 0]))
                fig.canvas.draw()
                plt.pause(0.05)
                happy = input('Happy? y or n: ')
                plt.close(1)
            paramDict['Box'] = boxTest
            """get the thresh"""
            happy = 'n'
            while happy == 'n': 
                threshTest = float(input("Enter the threshold estimate as percentage of max: "))
                fig, ax = plt.subplots(1,2, num = 1)
                mag = np.sqrt(np.sum(self.rawMag**2, axis = 0))
                ax[0].imshow(np.sum(mag, axis = 2))
                ax[1].imshow(np.sum(mag > np.amax(mag)*threshTest, axis = 2))
                fig.canvas.draw()
                plt.pause(0.05)
                happy = input('Happy? y or n: ')
                plt.close(1)
            paramDict['thresh'] = threshTest
        self.Params = paramDict
        rotatedCharge = np.zeros_like(self.rawCharge)
        rotatedMag = np.zeros_like(self.rawMag)
        for i in range(rotatedCharge.shape[2]): 
            rotatedCharge[...,i] = scipy.ndimage.rotate(self.rawCharge[...,i], int(self.Params[str(file[:3])]['Rot']), reshape = False)
            for j in range(rotatedMag.shape[0]): 
                rotatedMag[j, ..., i] = scipy.ndimage.rotate(self.rawMag[j, ..., i], int(self.Params[str(file[:3])]['Rot']), reshape = False)
        rec = {'charge': rotatedCharge, 
               'mag': rotatedMag}
        if t != None: 
            """Multi"""
            self.recDict.update({'{}'.format(t): rec})
            self.thetaDict.update({'{}'.format(t): r['theta_use'][0][::2]})
            self.projCalc.update({'{}'.format(t): r['proj_mag']})
            self.projMeas.update({'{}'.format(t): r['xmcd']})
        else:  
            """Standalone"""
            self.rec = rec
            self.temp = str(file[:3])
            self.generateMagneticArray(self.Params[str(file[:3])]['Box'], self.Params[str(file[:3])]['thresh'])
            self.magneticDomains()
            if 'theta_use' in list(r.keys()) == True: #FeRh experiment
                self.theta = r['theta_use'][0][::2]
                self.projCalc = r['proj_mag']
                self.projMeas = r['xmcd']
            elif 'theta_use' in list(r.keys()) != True and 'theta_cp' in list(r.keys()) == True: #PtPdMnSn
                self.theta = r['theta_cp']
                self.projMeas = (r['proj_out_all_cm_use'] - r['proj_out_all_cp_use'])/(r['proj_out_all_cm_use'] + r['proj_out_all_cp_use'])
                try: 
                    self.projCalc = r['proj_mag']
                except: 
                    print('No calculated projections found - carrying on without them')
            
        
        
    def generateMagneticArray(self, b, thresh, sampleArea = 20000, outline = False, t=None): 
        """Processed both the mag and charge by cropping to Params['box'] 
        and Thresholding to Params['thresh']"""
        if t != None: 
            m = self.recDict['{}'.format(t)]['mag']
            c = self.recDict['{}'.format(t)]['charge']
        else: 
            m = self.rec['mag']
            c = self.rec['charge']
        mNew = np.zeros(shape = (3, arraySize, arraySize, m.shape[3]))
        cNew = np.zeros(shape = (arraySize, arraySize, m.shape[3]))
        dims = [int(b[3]-b[2]), int(b[1]-b[0])]
        mNew[:, int((mNew.shape[1]-dims[0])/2):int((mNew.shape[1]+dims[0])/2), int((mNew.shape[2]-dims[1])/2):int((mNew.shape[2]+dims[1])/2), :] = m[:, b[2]:b[3], b[0]:b[1], :]
        cNew[int((mNew.shape[1]-dims[0])/2):int((mNew.shape[1]+dims[0])/2), 
             int((mNew.shape[2]-dims[1])/2):int((mNew.shape[2]+dims[1])/2), :] = c[b[2]:b[3], b[0]:b[1], :]
        mag = np.sqrt(mNew[0]**2 + mNew[1]**2 + mNew[2]**2)
        
        if outline == True: 
            outline = np.zeros_like(mag)
            mm = mag > 0
            print("Checking structure")
            for i in range(mag.shape[2]): 
                temp = binary_fill_holes(feature.canny(mm[...,i], 10))
                while np.sum(temp) < sampleArea: #20000 FeRh
                    print(i, np.sum(temp))
                    temp = binary_fill_holes(binary_dilation(temp))
                outline[...,i] = temp
                
        else: 
            outline = 0
        
        test = abs(mag) > thresh*np.amax(mag)
        mx = np.copy(mNew[0], order = "C")
        mx[~test] = 0

        my = np.copy(mNew[1], order = "C")
        my[~test] = 0

        mz = np.copy(mNew[2], order = "C")
        mz[~test] = 0
    
        mask = np.zeros_like(mz)
        mask[test] = 1
    
    
        if t != None:  
            self.charge.update({'{}'.format(t): c})
            self.magProcessed.update({'{}'.format(t): np.array([mx, my, mz])})
            self.magDict.update({'{}'.format(t): mag})
            self.magMasks.update({'{}'.format(t): np.array(mask)})
            self.chargeProcessed.update({'{}'.format(t):  cNew})
            self.sampleOutline.update({'{}'.format(t): outline})
        else: 
            self.charge = c
            self.magProcessed = np.array([mx, my, mz])
            self.mag = mag
            self.magMasks = np.array(mask)
            self.chargeProcessed = cNew
            self.sampleOutline = outline
            
    def volumeCalc(self, t = None):
        """Estimates the volume using the magMasks and the sample outline"""
        if t != None: 
            vol = {'volume': np.sum(self.magMasks['{}'.format(t)] == 1)/np.sum(self.sampleOutline['{}'.format(t)]), 
              'error': np.sum(self.magMasks['{}'.format(t)] == 1)/np.sum(self.sampleOutline['{}'.format(t)])*np.sqrt(1/np.sum(self.magMasks['{}'.format(t)] == 1) + 1/np.sum(self.sampleOutline['{}'.format(t)]))
              }
            self.volume.update({'{}'.format(t): vol})
        else: 
            vol = {'volume': np.sum(self.magMasks == 1)/np.sum(self.sampleOutline), 
              'error': np.sum(self.magMasks == 1)/np.sum(self.sampleOutline)*np.sqrt(1/np.sum(self.magMasks == 1) + 1/np.sum(self.sampleOutline))
              }
            self.volume.update({'{}'.format(t): vol})
            
    def calcCurl(self, t = None): 
        """Calculates magnetic curl"""
        if t != None: 
            m = np.copy(self.magProcessed['{}'.format(t)], order = "C")
        else: 
            m = np.copy(self.magProcessed, order = "C")
        curlx = np.gradient(m[2], axis = 0) - np.gradient(m[1], axis = 2)
        curly = -(np.gradient(m[2], axis = 1) - np.gradient(m[0], axis = 2)) 
        curlz = np.gradient(m[0], axis = 0) - np.gradient(m[1], axis = 1)

        
        curl = np.array([curlx, curly, curlz])
        if t != None: 
            self.curl.update({'{}'.format(t): curl})
        else: 
            self.curl = curl 
            
    def magneticDomains(self, t = None):
        """Finds area that point both +/- in each of the three magnetization direction"""
        if t != None: 
            m = np.copy(self.magProcessed['{}'.format(t)], order = "C")
            s = np.copy(self.sampleOutline['{}'.format(t)], order = "C")
        else: 
            m = np.copy(self.magProcessed, order = "C") 
            s = np.copy(self.sampleOutline, order = "C")
        mxPos = np.where(m[0] > 0)
        mxNeg = np.where(m[0] < 0)
        myPos = np.where(m[1] > 0)
        myNeg = np.where(m[1] < 0)
        mzPos = np.where(m[2] > 0)
        mzNeg = np.where(m[2] < 0)
        cMask = np.where(s == False)

        mxFin = np.zeros_like(m[0])
        mxFin[mxPos] = 1
        mxFin[mxNeg] = -1
        myFin = np.zeros_like(mxFin)
        myFin[myPos] = 1
        myFin[myNeg] = -1
        mzFin = np.zeros_like(mxFin)
        mzFin[mzPos] = 1
        mzFin[mzNeg] = -1

        mxFin[cMask] = np.nan
        myFin[cMask] = np.nan
        mzFin[cMask] = np.nan
        
        if t != None: 
            self.magMasks['{}'.format(t)][cMask] = -1
            self.magDomains.update({'{}'.format(t): np.array([mxFin, myFin, mzFin])})
        else: 
            self.magMasks[cMask] = -1
            self.magDomains = np.array([mxFin, myFin, mzFin])
            
    def saveParaview(self, savePath, t = None): 
        if t != None: 
            mx = self.magProcessed['{}'.format(t)][0]
            my = self.magProcessed['{}'.format(t)][1]
            mz = self.magProcessed['{}'.format(t)][2]
        else: 
            mx = self.magProcessed[0]
            my = self.magProcessed[1]
            mz = self.magProcessed[2]
            
        values = np.arange(mx.shape[0]*mx.shape[1]*mx.shape[2]).reshape(mx.shape)
        mesh = pv.UniformGrid()

        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        mesh.dimensions = np.array(values.shape) + 1

        # Edit the spatial reference
        mesh.origin = (-int(mx.shape[0]/2), -int(mx.shape[1]/2), -int(mx.shape[2]/2)) # The bottom left corner of the data set
        mesh.spacing = (1, 1, 1)  # These are the cell sizes along each axis

        # Add the data values to the cell data
        #mesh.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!
        mesh.cell_arrays["mx"] = mx.flatten(order="F")
        mesh.cell_arrays["my"] = my.flatten(order="F")
        mesh.cell_arrays["mz"] = mz.flatten(order="F")
        mesh.cell_arrays["mag"] = magDict['{}'.format(t)].flatten(order="F")
        mesh.cell_arrays["mag_vector"] = np.array([mx.flatten(order="F"), my.flatten(order="F"), mz.flatten(order="F")]).T
    

        os.chdir(savePath)
        if self.recDict != None: 
            mesh.save("{}K_{}_paraview.vtk".format(t, dateToSave(time = True)))
        else: 
            mesh.save("{}_paraview.vtk".format(dateToSave(time = True)))
            
    def CalculateVorticity(self, t = None): 
        """Calculates magnetic vorticity"""
        if t == None: 
            mx = self.magProcessed[0]/self.mag
            my = self.magProcessed[1]/self.mag
            mz = self.magProcessed[2]/self.mag
            m = np.array([mx, my, mz])
            v = np.zeros_like(m)
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    v[a] += E(a,b,c)*E(i,j,k)*m[i]*np.gradient(m[j], axis = b)*np.gradient(m[k], axis = c)
            self.vorticity = v
        else: 
            mx = self.magProcessed[t][0]/self.mag[t]
            my = self.magProcessed[t][1]/self.mag[t]
            mz = self.magProcessed[t][2]/self.mag[t]
            m = np.array([mx, my, mz])
            v = np.zeros_like(m)
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    v[a] += E(a,b,c)*E(i,j,k)*m[i]*np.gradient(m[j], axis = b)*np.gradient(m[k], axis = c)
            self.vorticity.update({t: v})
            
    
    def QuiverPlotSingle(self, direction, sliceNo, saveName = None, savePath = None): 
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'twilight_shifted'
        """Component = -1 gives the masks"""
        arr = {}
        quiverComps = {}
        if direction == 'x':
            shape = []
            comps = {}
            c = (2,0)
            for i in c: 
                comps.update({'{}'.format(i): np.swapaxes(self.magProcessed[i, sliceNo], 0,1)})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)

            scale = 0.3*200/shape[:,0]
        elif direction == 'y': 
            shape = []
            comps = {}
            c = (2,0)
            for i in c: 
                comps.update({'{}'.format(i): np.swapaxes(self.magProcessed[i,:,  sliceNo, :], 0,1)})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)
            
            scale = 0.3*200/shape[:,0]
        elif direction == 'z':
            shape = []
            comps = {}
            c = (0,1)
            for i in c: 
                    comps.update({'{}'.format(i): self.magProcessed[i, ..., sliceNo]})
            shape.append(1)
            shape = np.array(shape)
            scale = shape


        fig, ax = plt.subplots(figsize = (12,6), sharex = 'all')

         # Replace with plt.savefig if you want to save a file
        quiverKeys = list(comps.keys())
        mx = -comps['{}'.format(quiverKeys[0])]
        my = comps['{}'.format(quiverKeys[1])]
        m = abs(mx) > 0 
        scale2 = 2e-1*np.nanmean(abs(mx[np.nonzero(mx)]))
        x,y = np.meshgrid(np.arange(mx.shape[1]),
                          np.arange(mx.shape[0]))
        c = np.arctan2(my,mx)*m
        x[~m] = 0
        y[~m] = 0
        ax.imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
        ax.quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                       mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                       width = 0.005, scale = scale2, color = 'w', 
                      scale_units='dots') 
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_aspect(scale[i])
    
        display_axes = fig.add_axes([0.05,0.05,0.05,0.05], projection='polar')
        #display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!

        norm = mpl.colors.Normalize(-np.pi, np.pi)
        if direction == 'z':
            ll, bb, ww, hh = ax.get_position().bounds
            display_axes.set_position([1.7*ll, bb, 0.2*ww, 0.2*hh])
        else: 
            ll, bb, ww, hh = ax.get_position().bounds
            display_axes.set_position([ll*2, 0.95*bb, 0.5*ww, 0.5*hh])
        quant_steps = 2056
        cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap(cmap,quant_steps),
                                       norm=norm,
                                       orientation='horizontal')

        cb.outline.set_visible(False)                                 
        display_axes.set_axis_off()
        plt.tight_layout()
        if saveName != None:
            here = os.getcwd()
            os.chdir(savePath)
            fig.savefig('{}.svg'.format(saveName), dpi=1200)
            os.chdir(here)
                                    
                