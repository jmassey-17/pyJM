# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:50:25 2022

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
import matplotlib.pyplot as plt


from pyJM.BasicFunctions import *   
      
class Lamni(): 
    """
    A class that load, trim and process outputs of the Donnelly 3D xmcd 
    reconstruction algarithm.
 
    Attributes
    ----------
    file : str
        file to load.
    homedir : str
        directory where file sits.
    paramDict : dict
        dictionary of parameters for loading process. Can be none
    arraySize : int, optional
        size of output arrays . The default is 200.
    t : str, optional
        identified for use in lamniMulti. The default is None.
    """
    
    def __init__(self, fileToLoad, paramDict, arraySize = 200, t=None):
        """
        Initializes the Lamni object

        Parameters
        ----------
        file : str
            file to load.
        homedir : str
            directory where file sits.
        paramDict : dict
            dictionary of parameters for loading process. Can be none
        arraySize : int, optional
            size of output arrays . The default is 200.
        t : str, optional
            identified for use in lamniMulti. The default is None.

        Returns
        -------
        None.

        """
            
        """Load in the *mat file from the reconstructions"""
      
        r = scipy.io.loadmat(fileToLoad)
        
        """Crop it so only the non-masked elements in the mag are taken forward"""
        temp = r['mx_out']
        here = np.where(np.mean(temp, axis = (0,1)) != 0 )
        rec_charge = r['abs_out'][...,here[0]]
        rec_mag = np.array([r['mx_out'][...,here[0]], 
                            r['mz_out'][...,here[0]], # matlab out has mz as my
                            r['my_out'][...,here[0]]])
        
        """Extraction of temperature value - specific to FeRh Laminogrpahy experiment
        Change if required 
        """
        if t == None: 
            self.t = str(fileToLoad.split('\\')[4][:3])
        else: 
            self.t = t
            
        self.rawCharge = rec_charge 
        self.rawMag = rec_mag
        self.arraySize = arraySize
    
        """Identify the different parameters if doing for the first time."""
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
                cropTest = scipy.ndimage.rotate(abs(self.rawMag[0,...,0]), paramDict['Rot'], reshape = False)
                ax[0].imshow(cropTest)
                ax[1].imshow(cropTest[boxTest[2]:boxTest[3], boxTest[0]:boxTest[1]])
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
                mag = np.sqrt(np.sum(abs(self.rawMag**2), axis = 0))
                mag = scipy.ndimage.rotate(mag, paramDict['Rot'], reshape = False)
                mag = mag[paramDict['Box'][2]:paramDict['Box'][3], paramDict['Box'][0]:paramDict['Box'][1], :]
                ax[0].imshow(np.sum(abs(mag), axis = 2))
                ax[1].imshow(np.sum(abs(mag) > np.amax(abs(mag))*threshTest, axis = 2))
                fig.canvas.draw()
                plt.pause(0.05)
                happy = input('Happy? y or n: ')
                plt.close(1)
            paramDict['thresh'] = threshTest
            self.Params = {self.t: paramDict}
        else: 
            self.Params = paramDict
        
        rotatedCharge = np.zeros_like(self.rawCharge)
        rotatedMag = np.zeros_like(self.rawMag)
        for i in range(rotatedCharge.shape[2]): 
            rotatedCharge[...,i] = scipy.ndimage.rotate(self.rawCharge[...,i], int(self.Params[self.t]['Rot']), reshape = False)
            for j in range(rotatedMag.shape[0]): 
                rotatedMag[j, ..., i] = scipy.ndimage.rotate(self.rawMag[j, ..., i], int(self.Params[self.t]['Rot']), reshape = False)
        rec = {'charge': rotatedCharge, 
               'mag': rotatedMag}
        if t != None: 
            """Multi"""
            self.recDict.update({'{}'.format(t): rec})
            self.thetaDict.update({'{}'.format(t): r['theta_use'][0][::2]})
            self.projCalc.update({'{}'.format(t): r['proj_mag']})
            self.projMeas.update({'{}'.format(t): r['xmcd']})
            self.projMeasCorrected.update({f'{t}': (r[ 'proj_out_all_cp_use'] - r[ 'proj_out_all_cm_use'])/((r[ 'proj_out_all_cp_use'] + r[ 'proj_out_all_cm_use']))})
        else:  
            """Standalone"""
            self.rec = rec
            self.temp = self.t
            self.generateMagneticArray()
            if 'theta_use' in list(r.keys()) == True: #FeRh experiment
                self.theta = r['theta_use'][0][::2]
                self.projCalc = r['proj_mag']
                self.projMeas = r['xmcd']
                self.projMeasCorrected = (r[ 'proj_out_all_cp_use'] - r[ 'proj_out_all_cm_use'])/((r[ 'proj_out_all_cp_use'] + r[ 'proj_out_all_cm_use']))
            elif 'theta_use' in list(r.keys()) != True and 'theta_cp' in list(r.keys()) == True: #PtPdMnSn
                self.theta = r['theta_cp']
                self.projMeas = (r['proj_out_all_cm_use'] - r['proj_out_all_cp_use'])/(r['proj_out_all_cm_use'] + r['proj_out_all_cp_use'])
                try: 
                    self.projCalc = r['proj_mag']
                except: 
                    print('No calculated projections found - carrying on without them')
        
            
        
        
    def generateMagneticArray(self, outline = True, t=None): 
        """
        Performs the processing of the charge and magnetic arrays.
        Generates the sample outline mask, the AF/FM masks
        

        Parameters
        ----------
        thresh : float
           % threshold against which the magnetization magnitude is compared against .
        arraySize : int
            size of output array.
        outline : bool, optional
            calculate outline of sample. The default is True.
        t : str, optional
            identifier used with lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        
        if t == None: 
            m = self.rec['mag']
            c = self.rec['charge']
            b = self.Params[self.t]['Box']
            thresh = self.Params[self.t]['thresh']
        else: 
            m = self.recDict[f'{t}']['mag']
            c = self.recDict[f'{t}']['charge']
            b = self.Params[f'{t}']['Box']
            thresh = self.Params[f'{t}']['thresh']
        
        """Crop the arrays, need to be centered"""

        mNew = np.zeros(shape = (3, self.arraySize, self.arraySize, m.shape[3]))
        cNew = np.zeros(shape = (self.arraySize, self.arraySize, m.shape[3]))
        dims = [int(b[3]-b[2]), int(b[1]-b[0])]
        mNew[:, int((mNew.shape[1]-dims[0])/2):int((mNew.shape[1]+dims[0])/2), int((mNew.shape[2]-dims[1])/2):int((mNew.shape[2]+dims[1])/2), :] = m[:, b[2]:b[3], b[0]:b[1], :]
        cNew[int((mNew.shape[1]-dims[0])/2):int((mNew.shape[1]+dims[0])/2), 
             int((mNew.shape[2]-dims[1])/2):int((mNew.shape[2]+dims[1])/2), :] = c[b[2]:b[3], b[0]:b[1], :]
        mag = np.sqrt(mNew[0]**2 + mNew[1]**2 + mNew[2]**2)
        
        # Version from Feb 23
        # More accurate estimate of the actual sample
        if outline == True: 
            outline = mag > 0.01*np.amax(mag)
        else: 
            outline = 0
        
        
        test = abs(mag) > thresh*np.amax(abs(mag))
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
            
    def volumeCalc(self, box = None, t = None):
        """
        Method to calculate the volume of the sample by
        counting FM pixels/total sample pixels

        Parameters
        ----------
        box : list, optional
            box to use. The default is None.
        t : str, optional
            identifier for use in lamniMulti. The default is None.

        Returns
        -------
        None.

        """
       
        if t != None: 
            if box == None: 
                vol = {'volume': np.sum(self.magMasks['{}'.format(t)] == 1)/np.sum(self.sampleOutline['{}'.format(t)]), 
                  'error': np.sum(self.magMasks['{}'.format(t)] == 1)/np.sum(self.sampleOutline['{}'.format(t)])*np.sqrt(1/np.sum(self.magMasks['{}'.format(t)] == 1) + 1/np.sum(self.sampleOutline['{}'.format(t)]))
                  }
                
            else: 
                temp = self.magMasks['{}'.format(t)][box[2]:box[3], box[0]:box[1],:]
                vol = {'volume': np.sum(temp == 1)/temp.ravel().shape[0],
                  'error': np.sum(temp == 1)/np.sum(temp.ravel().shape[0])*np.sqrt(1/np.sum(temp == 1) + 1/np.sum(temp.ravel().shape[0]))
                  }
            self.volume.update({'{}'.format(t): vol})
        else: 
            if box == None: 
                vol = {'volume': np.sum(self.magMasks == 1)/np.sum(self.sampleOutline), 
                       'error': np.sum(self.magMasks == 1)/np.sum(self.sampleOutline)*np.sqrt(1/np.sum(self.magMasks == 1) + 1/np.sum(self.sampleOutline))
                       }
            else: 
                temp = self.magMasks[box[2]:box[3], box[0]:box[1],:]
                vol = {'volume': np.sum(temp == 1)/temp.ravel().shape[0],
                  'error': np.sum(temp == 1)/np.sum(temp.ravel().shape[0])*np.sqrt(1/np.sum(temp == 1) + 1/np.sum(temp.ravel().shape[0]))
                  }
            self.volume.update({'{}'.format(t): vol})
            
    def calcCurl(self, t = None): 
        """
        calculates magnetic curl

        Parameters
        ----------
        t : str, optional
            identifier for use with lamnimulti. The default is None.

        Returns
        -------
        None.

        """

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
            
    def saveParaview(self, savePath, t = None): 
        """
        Saves .vtk file for use in paraview. 
        File contains: 
            "x": x-coords 
            "y": y-coords 
            "z": z-coords 
            
            "c": charge array
            
            "mx": x-component of magnetization
            "my": y-component of magnetization
            "mz": z-component of magnetization
            "mag": magnitude of magnetization
            "mag_vector": magnetization vector
            
            'AF': AF regions = AF.flatten(order = "F")

        Parameters
        ----------
        savePath : str
            file path for saved file, should include directory.
        t : str, optional
            identifier for use in lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        
        import pyvista as pv
        if t != None: 
            mx = self.magProcessed[f'{t}'][0]
            my = self.magProcessed[f'{t}'][1]
            mz = self.magProcessed[f'{t}'][2]
            AF = self.magMasks[f'{t}'] == 0
            
            c = self.chargeProcessed[f'{t}']
            
            filtered = 0
            try: 
                getattr(self, 'filtered')
                filtered = 1
            except: 
                pass
            if filtered != 0: 
                mxFiltered = self.filtered['{}'.format(t)][0]
                myFiltered = self.filtered['{}'.format(t)][1]
                mzFiltered = self.filtered['{}'.format(t)][2]
        else: 
            mx = self.magProcessed[0]
            my = self.magProcessed[1]
            mz = self.magProcessed[2]
            AF = self.magMasks == 0
            filtered = 0
            try: 
                getattr(self, 'filtered')
                filtered = 1
            except: 
                pass
            if filtered != 0: 
                mxFiltered = self.filtered[0]
                myFiltered = self.filtered[1]
                mzFiltered = self.filtered[2]
            
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
        x,y,z = np.meshgrid(np.arange(mx.shape[0]),np.arange(mx.shape[1]),np.arange(mx.shape[2]))
        
        mesh.cell_arrays["x"] = x.flatten(order="F")
        mesh.cell_arrays["y"] = y.flatten(order="F")
        mesh.cell_arrays["z"] = z.flatten(order="F")
        
        mesh.cell_arrays["c"] = c.flatten(order="F")
        
        mesh.cell_arrays["mx"] = mx.flatten(order="F")
        mesh.cell_arrays["my"] = my.flatten(order="F")
        mesh.cell_arrays["mz"] = mz.flatten(order="F")
        
        
        mag = np.sqrt(mx**2 + my**2 + mz**2)
        mesh.cell_arrays["mag"] = mag.flatten(order="F")
        mesh.cell_arrays["mag_vector"] = np.array([mx.flatten(order="F"), my.flatten(order="F"), mz.flatten(order="F")]).T
        
        mesh.cell_arrays['AF'] = AF.flatten(order = "F")
        
        if filtered != 0: 
            mesh.cell_arrays["filteredMx"] = mxFiltered.flatten(order="F")
            mesh.cell_arrays["filteredMy"] = myFiltered.flatten(order="F")
            mesh.cell_arrays["filteredMz"] = mzFiltered.flatten(order="F")
            magFiltered = np.sqrt(mxFiltered**2 + myFiltered**2 + mzFiltered**2)
            mesh.cell_arrays["mag"] = magFiltered.flatten(order="F")
            mesh.cell_arrays["mag_vector"] = np.array([mxFiltered.flatten(order="F"), myFiltered.flatten(order="F"), mzFiltered.flatten(order="F")]).T
    

        os.chdir(savePath)
        mesh.save("{}K_{}_paraview.vtk".format(t, dateToSave()))
        
            
    def CalculateVorticity(self, attribute, t = None):
        """
        Calculates magnetic vorticity on a given attribute

        Parameters
        ----------
        attribute : str
            attribute to use in calculation.
        t : str, optional
            for use in lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        if t == None: 
            array = getattr(self, attribute)
            arrayMag = np.sqrt(array[0]**2 + array[1]**2 + array[2]**2)
            mx = array[0]/arrayMag
            my = array[1]/arrayMag
            mz = array[2]/arrayMag
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
            self.vorticityMag = np.sqrt(np.sum(v**2, axis = 0))
            div = 0
            for i in (0,1,2):
                div += np.gradient(v[i], axis = i)
            self.vorticityDivergence = div
        else:
            array = getattr(self, attribute)[t]
            arrayMag = np.sqrt(array[0]**2 + array[1]**2 + array[2]**2)
            mx = array[0]/arrayMag
            my = array[1]/arrayMag
            mz = array[2]/arrayMag
            m = np.array([mx, my, mz])
            v = np.zeros_like(m)
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    v[a] += E(a,b,c)*E(i,j,k)*m[i]*np.gradient(m[j], axis = b)*np.gradient(m[k], axis = c)
            div = 0
            for i in (0,1,2):
                div += np.gradient(v[i], axis = i)
            self.vorticity.update({t:{'raw': v, 
                                      'mag': np.sqrt(np.sum(v**2, axis = 0)), 
                                      'div': div}})
            
    def filterAttribute(self, attribute, sigma, t = None):
        """
        
        will apply gaussian filter of width sigma to self.attribute
        and returns it to self.filtered['attribute']
        
        Parameters
        ----------
        attribute : str
            attribute to filter.
        sigma : float
            size of gaussian function to filter by.
        t : str, optional
            identifier for compatibility with laniMulti. The default is None.

        Returns
        -------
        None.

        """
        
        from scipy.ndimage import gaussian_filter
        if t == None: 
            array = np.copy(getattr(self, attribute))
            filtered = np.zeros_like(array)
            for k in range(array.shape[3]): 
                filtered[ ..., k] = np.array([gaussian_filter(array[0,...,k], sigma), 
                                              gaussian_filter(array[1,...,k], sigma), 
                                              gaussian_filter(array[2,...,k], sigma)])
                                              
            self.filtered = filtered
        else: 
            array = getattr(self, attribute)[t]
            filtered = np.zeros_like(array)
            for k in range(array.shape[3]): 
                filtered[ ..., k] = np.array([gaussian_filter(array[0,...,k], sigma), 
                                              gaussian_filter(array[1,...,k], sigma), 
                                              gaussian_filter(array[2,...,k], sigma)])
            self.filtered.update({t: filtered})
            
    def QuiverPlotSingle(self, attribute, direction, sliceNo, xinterval, yinterval, scale2 = 0.0001, pos = [2, 1, 0.5, 0.5], saveName = None, savePath = None): 
        """
        
        plots quiver plot for attribute field
        
        Parameters
        ----------
        attribute : str
            attribute to plot
        direction : str
            direction of plane to plot
        sliceNo : int
            sliceNo to plot.
        xinterval : int
            interval over which the arrows are plotted in x-direction.
        yinterval : int
            interval over which the arrows are plotted in y-direction.
        scale2 : float, 
            Scale of the arrows in teh quiver plot. The default is 0.0001.
        pos : list, optional
            multiplier for position of box. The default is [2, 1, 0.5, 0.5].
        saveName : str, optional
            saveName for file. The default is None.
        savePath : str, optional
            save destination for figure. The default is None.

        Returns
        -------
        None.

        """
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'twilight_shifted'
        """Component = -1 gives the masks"""
        arr = {}
        quiverComps = {}
        if direction == 'x':
            xinterval = 1
            yinterval = 5
            shape = []
            comps = {}
            c = (2,0)
            for i in c: 
                comps.update({'{}'.format(i): np.swapaxes(getattr(self, attribute)[i, sliceNo], 0,1)})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)

            scale = 0.3*200/shape[:,0]
        elif direction == 'y': 
            xinterval = 1
            yinterval = 5
            shape = []
            comps = {}
            c = (2,0)
            for i in c: 
                comps.update({'{}'.format(i): np.swapaxes(getattr(self, attribute)[i,:,  sliceNo, :], 0,1)})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)
            
            scale = 0.3*200/shape[:,0]
        elif direction == 'z':
            xinterval = 6
            yinterval = 6
            shape = []
            comps = {}
            c = (0,1)
            for i in c: 
                    comps.update({'{}'.format(i): getattr(self, attribute)[i, ..., sliceNo]})
            shape.append(1)
            shape = np.array(shape)
            scale = shape


        fig, ax = plt.subplots(figsize = (12,6), sharex = 'all')

         # Replace with plt.savefig if you want to save a file
        quiverKeys = list(comps.keys())
        mx = comps['{}'.format(quiverKeys[0])]
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
            display_axes.set_position([pos[0]*ll, pos[1]*bb, pos[2]*ww, pos[3]*hh])
        else: 
            ll, bb, ww, hh = ax.get_position().bounds
            display_axes.set_position([pos[0]*ll, pos[1]*bb, pos[2]*ww, pos[3]*hh])
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
            
    def plotVectorField(self, field, t = None, box = None,  inplaneSkip = 0, outofplaneSkip = 0, saveName = None):
        """
        plots vector field using pyvista

        Parameters
        ----------
        field : str
            field to plot.
        t : str, optional
            identifier for use with lamniMulti. The default is None.
        box : list, optional
            box to show. The default is None.
        inplaneSkip : int, optional
            period over which to skip points in plane to reduce arrows. The default is 0.
        outofplaneSkip : int, optional
            period over which to skip points in plane to reduce arrows. The default is 0.
        saveName : str, optional
            file to save, should include directory if specific directory desired. The default is None.

        Returns
        -------
        None.

        """
        
        
        import pyvista as pv
        if t == None: 
            if box == None:
                f = getattr(self, field)
            else: 
                f = getattr(self, field)[:, box[2]:box[3], box[0]:box[1], :]
        else:  
            if box == None:
                f = getattr(self, field)[t]
            else: 
                f = getattr(self, field)[t][:, box[2]:box[3], box[0]:box[1], :]
                
        
        vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
        if inplaneSkip != 0: 
            vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
        if outofplaneSkip != 0: 
            vector_field = vector_field[:,..., ::outofplaneSkip]
            
        _, nx, ny, nz = vector_field.shape
        size = vector_field[0].size

        origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
        mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)

        mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
        mesh['mz'] = mesh['vectors'][:, 2]

        # # remove some values for clarity
        num_arrows = mesh['vectors'].shape[0]
        rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                     replace=False)

        mesh['vectors'][rand_ints] = 0
        mesh['scalars'] = -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1])


        mesh['vectors'][rand_ints] = np.array([0, 0, 0])
        arrows = mesh.glyph(factor=2, geom=pv.Arrow())
        pv.set_plot_theme("document")
        p = pv.Plotter()
        p.add_mesh(arrows, scalars='scalars', lighting=False, cmap='twilight_shifted', clim = [-np.pi, np.pi], show_scalar_bar=False)
        #p.show_grid()
        #p.add_bounding_box()

        y_down = [(0, 80, 0),
                  (0, 0, 0),
                  (0, 0, -90)]
        p.show(cpos=y_down)
        
        if saveName != None: 
            mesh.save('{}.vtk'.format(saveName))
        
        
    def plotScalarField(self, field):
        """
        Plot scalar field using pyvista

        Parameters
        ----------
        field : str
            field to plot.

        Returns
        -------
        None.

        """
        import pyvista as pv
        scalar_field = getattr(self, field)
        nx, ny, nz = scalar_field.shape
        size = scalar_field[0].size
        
        origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
        mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
        
        mesh['scalars'] = scalar_field.flatten(order = "F")
        
        
        # # remove some values for clarity
        num_arrows = mesh['scalars'].shape[0]
        rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                     replace=False)
        
        pv.set_plot_theme("document")
        p = pv.Plotter()
        p.add_mesh(mesh, scalars=mesh['scalars'], lighting=False, cmap='twilight_shifted')
        p.show_grid()
        p.add_bounding_box()
        
        y_down = [(0, 80, 0),
                  (0, 0, 0),
                  (0, 0, -90)]
        p.show(cpos=y_down)
        
            
    def countDistribution(self, t = None): 
        """
        count the number of the pixels pointing in either x,y or z directions

        Parameters
        ----------
        t : str, optional
            identifier for use with lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        if t == None: 
            test = self.magProcessed/np.sqrt(np.sum(self.magProcessed**2, axis = 0))
            unique, counts = np.unique(np.argmax(test, axis = 0), return_counts=True)
            self.distribution = {'directions': unique, 
                                 'counts': counts}
        else: 
            test = self.magProcessed[t]/np.sqrt(np.sum(self.magProcessed[t]**2, axis = 0))
            unique, counts = np.unique(np.argmax(test, axis = 0), return_counts=True)
            self.distribution.update({t: {'directions': unique, 
                                 'counts': counts}})
            
    
                
    def domainAnalysis2(self,thresh = 1, t = None): 
        """
        will measure properties of FM and AF areas above thresh using 
        skiamge.measure.regionprops

        Parameters
        ----------
        thresh : int, optional
            area threshold above which things are considreed real. The default is 1.
        t : str, optional
            identifier used for lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        import pandas as pd
        from skimage.measure import label, regionprops_table
        
        if t == None: 
            """Volume"""
            im = label(self.magMasks)#)[25:175, 40:160, :])
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'image_filled')))
            cleaned = regions[regions['area'] > thresh]

            cleaned['top'] =  cleaned['bbox-2'] == 0
            cleaned['bottom'] = cleaned['bbox-5'] == im.shape[-1]
            fm = cleaned
            
            im = label(1 - self.magMasks)#[25:175, 40:160, :])
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'image_filled')))
            cleaned = regions[regions['area'] > thresh]

            cleaned['top'] =  cleaned['bbox-2'] == 0
            cleaned['bottom'] = cleaned['bbox-5'] == im.shape[-1]
            af = cleaned
            
            self.domains2 = {'fm': fm, 
                             'af': af}
            
            """layer by layer"""
            for i in range(self.magMasks.shape[-1]): 
                """FM"""
                print('Hard coded at the moment to remove influence of the suprious sample edges')
                image = self.magMasks[...,i]#[25:175,40:160,i]
                label_img = label(image)
                regions = pd.DataFrame(regionprops_table(label_img, properties = ('centroid',
                                                 'orientation',
                                                 'axis_major_length',
                                                 'axis_minor_length', 
                                                 'area', 
                                                 'bbox', 
                                                 'image_filled')))
                regions = regions[regions['area'] > thresh]
                regions['domainIdentifier'] = np.arange(len(regions))
                regions['slice'] = i*(145/self.magMasks.shape[-1])
                regions['temp'] = self.t
                if i == 0: 
                    fm_individual = regions
                else: 
                    fm_individual = fm_individual.append(regions, ignore_index = True)
                
                """AF"""
                image = 1-self.magMasks[...,i]#[25:175,40:160,i]
                label_img = label(image)
                regions = pd.DataFrame(regionprops_table(label_img, properties = ('centroid',
                                                 'orientation',
                                                 'axis_major_length',
                                                 'axis_minor_length', 
                                                 'area', 
                                                 'bbox', 
                                                 'image_filled')))
                regions = regions[regions['area'] > thresh]
                regions['domainIdentifier'] = np.arange(len(regions))
                regions['slice'] = i*(145/self.magMasks.shape[-1])
                regions['temp'] = self.t
                if i == 0: 
                    af_individual = regions
                else: 
                    af_individual = af_individual.append(regions, ignore_index = True)
                
            self.domains2individual = {'fm' : fm_individual, 
                                       'af' : af_individual}
        else: 
            im = label(self.magMasks[t])#[25:175, 40:160, :])
            image2 = self.magMasks[t]
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'coords',
                                             'image_filled')))
            cleaned = regions[regions['area'] > thresh]
            cleaned = cleaned[cleaned['area'] != np.sum(image2 == -1)]
            cleaned['top'] =  cleaned['bbox-2'] == 0
            cleaned['bottom'] = cleaned['bbox-5'] == im.shape[-1]
            cleaned['temp'] = t
            fm = cleaned
                
      
            im = label(1 - self.magMasks[t])#[25:175, 40:160, :])
            image2 = 1 - self.magMasks[t]
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox',
                                             'coords',
                                             'image_filled')))
            cleaned = regions[regions['area'] > thresh]
            cleaned = cleaned[cleaned['area'] != np.sum(image2 == -1)]
            cleaned['top'] =  cleaned['bbox-2'] == 0
            cleaned['bottom'] = cleaned['bbox-5'] == im.shape[-1]
            cleaned['temp'] = t
            af = cleaned
           
            
            self.domains2.update({t: {'fm': fm, 
                             'af': af}})
            
            
            """layer by layer"""
            for i in range(self.magMasks[t].shape[-1]): 
                """FM"""
                image = self.magMasks[t][...,i]#[25:175,40:160,i]
                label_img = label(image)
                regions = pd.DataFrame(regionprops_table(label_img, properties = ('centroid',
                                                 'orientation',
                                                 'axis_major_length',
                                                 'axis_minor_length', 
                                                 'area', 
                                                 'bbox', 
                                                 'image_filled')))
                regions = regions[regions['area'] > thresh]
                regions['domainIdentifier'] = np.arange(len(regions))
                regions['slice'] = i*(145/self.magMasks[t].shape[-1])
                regions['temp'] = t
                
                if i == 0: 
                    fm_individual = regions
                else: 
                    fm_individual = fm_individual.append(regions, ignore_index = True)
                
                """AF"""
                image = 1-self.magMasks[t][...,i]#[25:175,40:160,i]
                label_img = label(image)
                regions = pd.DataFrame(regionprops_table(label_img, properties = ('centroid',
                                                 'orientation',
                                                 'axis_major_length',
                                                 'axis_minor_length', 
                                                 'area', 
                                                 'bbox', 
                                                 'image_filled')))
                regions = regions[regions['area'] > thresh]
                regions['domainIdentifier'] = np.arange(len(regions))
                regions['slice'] = i*(145/self.magMasks[t].shape[-1])
                regions['temp'] = t
                if i == 0: 
                    af_individual = regions
                else: 
                    af_individual = af_individual.append(regions, ignore_index = True)
                
            self.domains2individual.update({t: {'fm' : fm_individual, 
                                       'af' : af_individual}})
            
    def calcDistributions(self, t = None):
        """
        calculates FM/AF regions per layer

        Parameters
        ----------
        t : str, optional
            identifier for use with lamniMulti. The default is None.

        Returns
        -------
        None.

        """
        if t == None: 
            fm = np.zeros(shape = self.magMasks.shape[-1])
            af = np.zeros(shape = self.magMasks.shape[-1])
            fmerr = np.zeros(shape = self.magMasks.shape[-1])
            aferr = np.zeros(shape = self.magMasks.shape[-1])
            for i in range(self.magMasks.shape[-1]): 
                fm[i] = np.sum(self.magMasks[...,i] == 1)/np.sum(self.magMasks[...,i]>-1)
                fmerr[i] = np.sum(self.magMasks[...,i] == 1)/np.sum(self.magMasks[...,i]>-1)*np.sqrt((1/np.sum(self.magMasks[...,i] == 1)) + (1/np.sum(self.magMasks[...,i]>0)))
                af[i] = (np.sum(self.magMasks[...,i] == 0)/np.sum(self.magMasks[...,i]>-1))
                aferr[i] = (np.sum(self.magMasks[...,i] == 0)/np.sum(self.magMasks[...,i]>-1)*np.sqrt((1/np.sum(self.magMasks[...,i] == 1)) + (1/np.sum(self.magMasks[...,i]>0))))
            self.distributions = {'fm': [np.array(fm), np.array(fmerr)],
                                     'af': [np.array(af), np.array(aferr)]}
            print('Distribution calculated succesfully.')
        else: 
            fm = np.zeros(shape = self.magMasks[t].shape[-1])
            af = np.zeros(shape = self.magMasks[t].shape[-1])
            fmerr = np.zeros(shape = self.magMasks[t].shape[-1])
            aferr = np.zeros(shape = self.magMasks[t].shape[-1])
            for i in range(self.magMasks[t].shape[-1]): 
                fm[i] = np.sum(self.magMasks[t][...,i] == 1)/np.sum(self.magMasks[t][...,i]>-1)
                fmerr[i] = np.sum(self.magMasks[t][...,i] == 1)/np.sum(self.magMasks[t][...,i]>-1)*np.sqrt((1/np.sum(self.magMasks[t][...,i] == 1)) + (1/np.sum(self.magMasks[t][...,i]>0)))
                af[i] = (np.sum(self.magMasks[t][...,i] == 0)/np.sum(self.magMasks[t][...,i]>-1))
                aferr[i] = (np.sum(self.magMasks[t][...,i] == 0)/np.sum(self.magMasks[t][...,i]>-1)*np.sqrt((1/np.sum(self.magMasks[t][...,i] == 1)) + (1/np.sum(self.magMasks[t][...,i]>0))))
            self.distributions.update({t: {'fm': [np.array(fm), np.array(fmerr)],
                                     'af': [np.array(af), np.array(aferr)]}})
            print(f'Distribution calculated succesfully for {t} K')