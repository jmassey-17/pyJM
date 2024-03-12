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
import pyvista as pv


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
        
        """Define presets for pyvista images"""
        self.Define3DPresets()
                 
    
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
            print(t)
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
        
        
    def Define3DPresets(self): 
        """
        Defines preset dictionaries to use with the 3D plotter function

        Returns
        -------
        None.

        """
        self.presets = {'FM': {
                          # fields
                          'field': 'magProcessed',
                          'maskField': 'magMasks',
                          # plot options
                          'includeArrows': True,
                          'includeAF': False,
                          'includeOutline': True,
                          'includeScaleBar': False,
                          'includeBoundingBox': True,
                          'includeDirectionalKey': False,
                          'includeMagnetization': True,
                          'includeScaleBarMagnitude': False,
                          'randomRemoval': True,
                          # plot specifics
                           'colourBy': 'my',
                           'inplaneSkip': 3,
                           'outofplaneSkip': 0,
                           'box': [125,155,0, 200],
                           'zScale': 1,
                           'arrowScale': 4,
                          # figure options
                           'p': None,
                           'tArray': None,
                            't': '310', 
                            'camera': (0.0, 0.0, 125), 
                            'save': True,
                          }, 
                   
                   'AF': {
                       # fields
                       'field': 'magProcessed',
                       'maskField': 'magMasks',
                       # plot options
                       'includeArrows': True,
                       'includeAF': False,
                       'includeOutline': True,
                       'includeScaleBar': False,
                       'includeBoundingBox': True,
                       'includeDirectionalKey': False,
                       'includeMagnetization': True,
                       'includeScaleBarMagnitude': False,
                       'randomRemoval': True,
                       # plot specifics
                        'colourBy': 'my',
                        'inplaneSkip': 3,
                        'outofplaneSkip': 0,
                        'box': [36,66,0, 200],
                        'zScale': 1,
                        'arrowScale': 4,
                       # figure options
                        'p': None,
                        'tArray': None,
                         't': '330', 
                         'camera': (0.0, 0.0, 125), 
                         'save': True,
                          }, 
                   
                   'down': {
                       # fields
                       'field': 'magProcessed',
                       'maskField': 'magMasks',
                       # plot options
                       'includeArrows': True,
                       'includeAF': False,
                       'includeOutline': True,
                       'includeScaleBar': False,
                       'includeBoundingBox': True,
                       'includeDirectionalKey': False,
                       'includeMagnetization': True,
                       'includeScaleBarMagnitude': False,
                       'randomRemoval': True,
                       # plot specifics
                        'colourBy': 'my',
                        'inplaneSkip': 3,
                        'outofplaneSkip': 0,
                        'box': None,
                        'zScale': 1,
                        'arrowScale': 4,
                       # figure options
                        'p': pv.Plotter(shape = (2,3), border = False, lighting = 'three lights'),
                        'tArray': {'440': [0,0],
                                  '330': [0,1], 
                                  '300': [0,2],
                                  '375': [1,2],
                                  '335': [1,1], 
                                  '310': [1,0],
                                  },
                         't': None, 
                         'camera': (0.0, 0.0, 145), 
                         'save': True,
                          },
                   'side': {
                       # fields
                       'field': 'magProcessed',
                       'maskField': 'magMasks',
                       # plot options
                       'includeArrows': True,
                       'includeAF': True,
                       'includeOutline': True,
                       'includeScaleBar': False,
                       'includeBoundingBox': True,
                       'includeDirectionalKey': False,
                       'includeMagnetization': False,
                       'includeScaleBarMagnitude': False,
                       'randomRemoval': True,
                       # plot specifics
                        'colourBy': 'my',
                        'inplaneSkip': 3,
                        'outofplaneSkip': 0,
                        'box': [90,110,0,-1],
                        'zScale': 5,
                        'arrowScale': 4,
                       # figure options
                        'p': pv.Plotter(shape = (3,2), border = False, lighting = 'three lights'),
                        'tArray': {'440': [0,0],
                                    '330': [1,0], 
                                    '300': [2,0],
                                    '375': [2,1],
                                    '335': [1,1], 
                                    '310': [0,1],
                                    },
                         't': None, 
                         'camera': (0.0, -75, 0), 
                         'save': True,
                          },
                          }
        
    def plot3Dfields(self, view, field, maskField, 
                         includeArrows, includeAF, includeOutline, includeScaleBar, includeBoundingBox, includeDirectionalKey, includeMagnetization, includeScaleBarMagnitude, randomRemoval, 
                         colourBy, inplaneSkip, outofplaneSkip, box, zScale, arrowScale, 
                         p, tArray, t, camera, save):
            """
            

            Parameters
            ----------
            view : str
                name of preset.
            field : str
                field to image.
            maskField : str
                mask field to image .
            includeArrows : Bool
            includeAF : Bool
            includeOutline :Bool
            includeScaleBar :Bool
            includeBoundingBox : Bool
            includeDirectionalKey : Bool
            includeMagnetization : Bool
            includeScaleBarMagnitude : Bool
            randomRemoval : Bool
                Remove points randomly to help visualization.
            colourBy : str
                quantity to colour arrows by .
            inplaneSkip : int
                number of spins skipped in in plane direction.
            outofplaneSkip : int
                number of spins skipped in out of  plane direction.
            box : list
                box to visualize
            zScale : float
                scale of z direction compared to x,y.
            arrowScale : float
                size of arrow.
            p : pv.Plotter
                instantiation of the pv.Plotter.
            tArray : dict
                used to define what goes where in p.
            t : str
                t used for single images.
            camera : tuple
                camera position for use in p.
            save : Bool
                save image.

            Returns
            -------
            None.

            """
            
            cmaps = {'arrows': 'coolwarm', 
                     'outline': 'twilight_shifted', 
                     'magnitude': 'binary',
                     'af': 'viridis',
                     }
            
            fileName = f'{view}'
            if tArray != None: 
                for i, t in enumerate(list(getattr(self, field).keys())):
                    if box == None:
                        f = getattr(self, field)[t]
                    else: 
                        f = getattr(self, field)[t][:, box[2]:box[3], box[0]:box[1], :]
                         
                    vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
                    mag_field = np.sqrt(np.sum(f**2, axis = 0))/np.nanmax(np.sqrt(np.sum(f**2, axis = 0)))
                    vector_field[np.isnan(vector_field)] = 0
                    mag_field[np.isnan(mag_field)] = 0
                    if inplaneSkip != 0: 
                        vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
                        mag_field = mag_field[::inplaneSkip, ::inplaneSkip, :]
                    if outofplaneSkip != 0: 
                        vector_field = vector_field[:,..., ::outofplaneSkip]
                        mag_field = mag_field[..., ::outofplaneSkip]
                
                    _, nx, ny, nz = vector_field.shape
                    size = vector_field[0].size
                    offset = [-0.5, -0.5, -0.5]
                    
                    origin = (-(nx - 1) * 1 / 2 , -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
                    mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                    
                    mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
                    mesh['m'] = mag_field.T.reshape(size)
                    
                    mesh['mx'] = mesh['vectors'][:, 0]
                    mesh['my'] = mesh['vectors'][:, 1]
                    mesh['mz'] = mesh['vectors'][:, 2]
                    
                    #
                    
                    if randomRemoval: 
                        # remove some values for clarity
                        num_arrows = mesh['vectors'].shape[0]
                        rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                                      replace=False)
                        mesh['vectors'][rand_ints] = 0
                        
                    
                    colourByDictionary = {
                        'angle': {'values': -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1]), 
                                  'upperLimit': np.pi, 
                                  'lowerLimit': -np.pi},
                        'mx': {'values': mesh['vectors'][:, 0], 
                                  'upperLimit': 1, 
                                  'lowerLimit': -1},
                        'my': {'values': mesh['vectors'][:, 1], 
                                  'upperLimit': 1, 
                                  'lowerLimit': -1},
                        'mz': {'values': mesh['vectors'][:, 2], 
                                  'upperLimit': 1, 
                                  'lowerLimit': -1},
                        }
                
                    mesh['scalars'] = colourByDictionary[colourBy]['values']
                    
                    arrows = mesh.glyph(factor=arrowScale, geom=pv.Arrow())
                    
                    afAndOutline = 1 - getattr(self, maskField)[t]
                    o = 1 - getattr(self, 'sampleOutline')[t]
                    af = afAndOutline - (o)
                    if box != None: 
                        o = o[box[2]:box[3], box[0]:box[1], :]
                        af = af[box[2]:box[3], box[0]:box[1], :]
                            
                    scalar_field = af*1
                    scalar_field[af != True] = 0
                    scalar_field[af == True] = 250
                    
                    scalar_field2 = o
                    scalar_field3 = self.magDict[t]
                    m = np.amax(scalar_field3)
                    if box != None:
                        scalar_field3 = scalar_field3[box[2]:box[3], box[0]:box[1], :]
                            
                    if inplaneSkip != 0: 
                        scalar_field = scalar_field[::inplaneSkip, ::inplaneSkip, :]
                        scalar_field2 = scalar_field2[::inplaneSkip, ::inplaneSkip, :]
                        scalar_field3 = scalar_field3[::inplaneSkip, ::inplaneSkip, :]
                    if outofplaneSkip != 0: 
                        scalar_field = scalar_field[...,::outofplaneSkip]
                        scalar_field2 = scalar_field2[...,::outofplaneSkip]
                        scalar_field3 = scalar_field3[...,::outofplaneSkip]
                    nx, ny, nz = scalar_field.shape
                    size = scalar_field[0].size
                    
                    
                    
                    origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
                    mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                    mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                    mesh3 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                    mesh1['scalars'] = scalar_field.flatten(order = "F")
                    mesh2['scalars'] = scalar_field2.flatten(order = "F")
                    mesh3['scalars'] = scalar_field3.flatten(order = "F")
                
                
                    
                    pv.set_plot_theme("document")
                    p.background_color = "white"   
                    p.subplot(tArray[t][0], tArray[t][1])
                    opacity = [0,.7]
                    if view == 'down': 
                        p.camera_position = 'xy'
                        p.camera.roll = -90
                    p.camera.position = camera
                    #p.add_mesh(stream.tube(radius=0.1))
                    
                    if includeArrows:
                        p.add_mesh(arrows, scalars='scalars', cmap=cmaps['arrows'], clim = [colourByDictionary[colourBy]['lowerLimit'],colourByDictionary[colourBy]['upperLimit']], show_scalar_bar = False)
                        if i == 0: 
                            fileName = fileName + '_arrows_'
                    if includeScaleBar:
                        p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
                        if i == 0: 
                            fileName = fileName + '_scalebar_'
                    if includeAF and (np.sum(af) > 0):
                        p.add_volume(mesh1, scalars='scalars', opacity = [0,.8],  cmap=cmaps['af'], show_scalar_bar = False)
                        if i == 0: 
                            fileName = fileName + '_AF_'
                    if includeOutline:
                        p.add_volume(mesh2, scalars='scalars', cmap=cmaps['outline'], opacity = opacity, show_scalar_bar = False)
                        if i == 0: 
                            fileName = fileName + '_outline_'
                    if includeBoundingBox:
                        p.add_bounding_box()
                        if i == 0: 
                           fileName = fileName + '_boundingBox_'
                    if includeDirectionalKey: 
                        p.add_axes(labels_off = True, line_width = 5)
                        if i == 0: 
                            fileName = fileName + '_key_'
                    if includeMagnetization:
                        p.add_volume(mesh3, scalars='scalars', cmap=cmaps['magnitude'], clim = [m/6, m/4], opacity = [0,.1], show_scalar_bar = False)
                        # [np.amax(mesh3['scalars'])/6, np.amax(mesh3['scalars'])/3]for down
                        if i == 0: 
                            fileName = fileName + '_magnitude_'
                    if includeScaleBarMagnitude:
                        p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
                        if i == 0: 
                            fileName = fileName + '_mBar_'
                if save: 
                    p.save_graphic(f'{fileName}.svg')
                p.show()
            else: 
            
                if box == None:
                    f = getattr(self, field)[t]
                else: 
                    f = getattr(self, field)[t][:, box[2]:box[3], box[0]:box[1], :]
                         
                vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
                mag_field = np.sqrt(np.sum(f**2, axis = 0))/np.nanmax(np.sqrt(np.sum(f**2, axis = 0)))
                vector_field[np.isnan(vector_field)] = 0
                mag_field[np.isnan(mag_field)] = 0
                if inplaneSkip != 0: 
                    vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
                    mag_field = mag_field[::inplaneSkip, ::inplaneSkip, :]
                if outofplaneSkip != 0: 
                    vector_field = vector_field[:,..., ::outofplaneSkip]
                    mag_field = mag_field[..., ::outofplaneSkip]
            
                _, nx, ny, nz = vector_field.shape
                size = vector_field[0].size
                
                
                origin = (-(nx - 1) * 1 / 2 , -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
                mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                
                mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
                mesh['m'] = mag_field.T.reshape(size)
                
                mesh['mx'] = mesh['vectors'][:, 0]
                mesh['my'] = mesh['vectors'][:, 1]
                mesh['mz'] = mesh['vectors'][:, 2]
                
                #
                
                if randomRemoval: 
                    # remove some values for clarity
                    num_arrows = mesh['vectors'].shape[0]
                    rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                                  replace=False)
                    mesh['vectors'][rand_ints] = 0
                    
                
                colourByDictionary = {
                    'angle': {'values': -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1]), 
                              'upperLimit': np.pi, 
                              'lowerLimit': -np.pi},
                    'mx': {'values': mesh['vectors'][:, 0], 
                              'upperLimit': 1, 
                              'lowerLimit': -1},
                    'my': {'values': mesh['vectors'][:, 1], 
                              'upperLimit': 1, 
                              'lowerLimit': -1},
                    'mz': {'values': mesh['vectors'][:, 2], 
                              'upperLimit': 1, 
                              'lowerLimit': -1},
                    }
            
                mesh['scalars'] = colourByDictionary[colourBy]['values']
                
                arrows = mesh.glyph(factor=arrowScale, geom=pv.Arrow())
                maskField = 'magMasks'
                
                if t == None: 
                    afAndOutline = 1 - getattr(self, maskField)
                    o = 1- getattr(self, 'sampleOutline')
                    af = afAndOutline - (o)
                    if box != None: 
                        o = o[box[2]:box[3], box[0]:box[1], :]
                        af = af[box[2]:box[3], box[0]:box[1], :]
                else:  
                    afAndOutline = 1 - getattr(self, maskField)[t]
                    o = 1 - getattr(self, 'sampleOutline')[t]
                    af = afAndOutline - (o)
                    if box != None: 
                        o = o[box[2]:box[3], box[0]:box[1], :]
                        af = af[box[2]:box[3], box[0]:box[1], :]
                        
                scalar_field = af*1
                scalar_field[af != True] = 0
                scalar_field[af == True] = 250
                
                scalar_field2 = o
                scalar_field3 = self.magDict[t]
                m = np.amax(scalar_field3)
                if box != None:
                    scalar_field3 = scalar_field3[box[2]:box[3], box[0]:box[1], :]
                        
                if inplaneSkip != 0: 
                    scalar_field = scalar_field[::inplaneSkip, ::inplaneSkip, :]
                    scalar_field2 = scalar_field2[::inplaneSkip, ::inplaneSkip, :]
                    scalar_field3 = scalar_field3[::inplaneSkip, ::inplaneSkip, :]
                if outofplaneSkip != 0: 
                    scalar_field = scalar_field[...,::outofplaneSkip]
                    scalar_field2 = scalar_field2[...,::outofplaneSkip]
                    scalar_field3 = scalar_field3[...,::outofplaneSkip]
                nx, ny, nz = scalar_field.shape
                size = scalar_field[0].size
                
                
                
                origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
                mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                mesh3 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale), origin)
                mesh1['scalars'] = scalar_field.flatten(order = "F")
                mesh2['scalars'] = scalar_field2.flatten(order = "F")
                mesh3['scalars'] = scalar_field3.flatten(order = "F")
            
                p = pv.Plotter()
                pv.set_plot_theme("document")
                p.background_color = "white"   
                p.camera_position = 'xy'
                p.camera.roll = -90
                p.camera.position = camera
                
                opacity = [0,.7]
                
            
                if includeArrows:
                    p.add_mesh(arrows, scalars='scalars', cmap=cmaps['arrows'], clim = [colourByDictionary[colourBy]['lowerLimit'],colourByDictionary[colourBy]['upperLimit']], show_scalar_bar = False)
                    fileName = fileName + '_arrows_'
                if includeScaleBar:
                    p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
                    fileName = fileName + '_scalebar_'
                if includeAF and (np.sum(af) > 0):
                    p.add_volume(mesh1, scalars='scalars', opacity = [0,.8],  cmap=cmaps['af'], show_scalar_bar = False)
                    fileName = fileName + '_AF_'
                if includeOutline:
                    p.add_volume(mesh2, scalars='scalars', cmap=cmaps['outline'], opacity = opacity, show_scalar_bar = False)
                    fileName = fileName + '_outline_'
                if includeBoundingBox:
                    p.add_bounding_box()
                    fileName = fileName + '_boundingBox_'
                if includeDirectionalKey: 
                    p.add_axes(labels_off = True, line_width = 5)
                    fileName = fileName + '_key_'
                if includeMagnetization:
                    p.add_volume(mesh3, scalars='scalars', cmap=cmaps['magnitude'], clim = [m/6, m/4], opacity = [0,.1], show_scalar_bar = False)
                    # [np.amax(mesh3['scalars'])/6, np.amax(mesh3['scalars'])/3]for down
                    fileName = fileName + '_magnitude_'
                if includeScaleBarMagnitude:
                    p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
                    fileName = fileName + '_mBar_'
                if save: 
                    p.save_graphic(f'{fileName}.svg')
                p.show()
    
                          

