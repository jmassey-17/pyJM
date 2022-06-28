# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:42:59 2022

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


#from pyJM.ThreeDimensional.Lamni import Lamni
from pyJM.ThreeDimensional.LamniMulti import LamniMulti

class LamniOutput(LamniMulti): 
    def __init__(self, homedir, paramDict):
        import matplotlib.pyplot as plt
        super().__init__(homedir, paramDict)
        heatScans = []
        coolScans = []
        for key in list(self.Params.keys()):
            if self.Params[key]['H or C'] == "H": 
                heatScans.append(key)
            else:
                coolScans.append(key)
        self.heatScans = heatScans
        self.coolScans = coolScans
    
    def MeasuredCalculatedComparison(self, angle2show, saveName = None, savePath = None): 
        """Need the comparison figure in here for the calculated and measured """

        """Heat and cool for a given angle"""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        
        
        angles = np.round(self.thetaDict['300'], 0)
        fig, ax = plt.subplots(2,3, constrained_layout=True)
        fig.suptitle('Measured XMCD for {} Degrees'.format(angle2show), fontsize=16)
        cmap = 'twilight_shifted'
        i = 0
        for h in list(self.coolScans): 
            temp, index = np.unique(np.round(self.thetaDict['{}'.format(h)], 0), return_index = True)
            xmcd = self.projMeas['{}'.format(h)][...,index]
            calc = self.projCalc['{}'.format(h)][...,index]
            if temp[-1] == 360: 
                temp = np.roll(temp, 1)
                temp[0] = 0
            here = np.where(temp == angle2show)[0] + self.Params[h]['thetaoffset'] 
            if here >= xmcd.shape[2]: 
                here = here-xmcd.shape[2]
            vmin = -0.05
            vmax = 0.05
            bar = ax[0 , i].imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[1 , i].imshow(calc[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
            i += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        
        
        if saveName != None: 
            saveName = saveName + dateToSave(time = True)
            he = os.getcwd()
            os.chdir(savePath)
            plt.savefig('{}_{}_Measured.svg'.format(saveName, h), dpi=1200)
            os.chdir(he)
            
    def MultiAngleMeasCalcComp(self, angles2show, kw = 'heat'):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'twilight_shifted'
        if kw == 'heat':
            scans = self.heatScans
        else: 
            scans = self.coolScans
        angles = np.round(self.thetaDict['300'], 0)
        fig, ax = plt.subplots(len(scans), len(angles2show), constrained_layout=True)
        vmin = -0.05
        vmax = 0.05
        i = 0
        for h in list(scans): 
            temp, index = np.unique(np.round(self.thetaDict['{}'.format(h)], 0), return_index = True)
            xmcd = self.projMeas['{}'.format(h)][...,index]
            if temp[-1] == 360: 
                temp = np.roll(temp, 1)
                temp[0] = 0
            j = 0
            for angle in angles2show: 
                here = np.where(temp == angle)[0] + self.Params[h]['thetaoffset'] 
                if here >= xmcd.shape[2]: 
                    here = here-xmcd.shape[2]
                bar = ax[i, j].imshow(xmcd[...,here],  cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i, j].axis('off')
                #if i == 0: 
                #    ax[i, j].set_title('{}{}'.format(angle, degree_sign))
                j += 1
            i += 1
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show()

    def HeatCoolForGivenAngle(self, angle2show):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'twilight_shifted'
        angles = np.round(self.thetaDict['300'], 0)
        fig, ax = plt.subplots(2,3, constrained_layout=True)
        fig.suptitle('Measured XMCD for {} Degrees'.format(angle2show), fontsize=16)
        i = 0
        for h in list(self.heatScans): 
            temp, index = np.unique(np.round(self.thetaDict['{}'.format(h)], 0), return_index = True)
            xmcd = self.projMeas['{}'.format(h)][...,index]
            if temp[-1] == 360: 
                temp = np.roll(temp, 1)
                temp[0] = 0
            here = np.where(temp == angle2show)[0] + self.Params[h]['thetaoffset']
            if here >= xmcd.shape[2]: 
                here = here-xmcd.shape[2]
            vmin = -0.05
            vmax = 0.05
            if i < 3:
                bar = ax[i //3 , i].imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i // 3, i].axis('off')
                ax[i // 3, i].set_title('{} K'.format(h))

            elif i >= 3: 
                bar = ax[i // 3, i-3].imshow(xmcd[...,here],  cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i // 3, i-3].axis('off')
                ax[i // 3, i].set_title('{} K'.format(h))
            i += 1
        for c in list(self.coolScans): 
            temp, index = np.unique(np.round(self.thetaDict['{}'.format(c)], 0), return_index = True)
            xmcd = self.projMeas['{}'.format(c)][...,index]
            if temp[-1] == 360: 
                temp = np.roll(temp, 1)
                temp[0] = 0
            here = np.where(temp == angle2show)[0] + self.Params[c]['thetaoffset']
            if here >= xmcd.shape[2]: 
                here = here-xmcd.shape[2]
            vmin = -0.05
            vmax = 0.05
            if i < 3:
                bar = ax[i //3 , i].imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i // 3, i].axis('off')
                ax[i // 3, i].set_title('{} K'.format(c))

            elif i >= 3: 
                bar = ax[i // 3, i-3].imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i // 3, i-3].axis('off')
                ax[i // 3, i-3].set_title('{} K'.format(c))
            i += 1
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show()
        
    def ScansForVariousAngles(self, angles2show, kw = 'heat', fileName = None, savePath = None):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'bwr'
        if kw == 'heat': 
            scans = self.heatScans
        else: 
            scans = self.coolScans
        angles = np.round(self.thetaDict['300'], 0)
        fig, ax = plt.subplots(len(angles2show), len(scans), figsize = (8,6))
        vmin = -0.05
        vmax = 0.05
        i = 0
        cols = ['{} K'.format(col) for col in scans]
        rows = ['{}'.format(row) for row in angles2show]

        for a, col in zip(ax[0], cols):
            a.set_title(col)

        for a, row in zip(ax[:,0], rows):
            a.set_ylabel(row, rotation=0, size='large')

        for angle in angles2show: 
            j = 0
            for h in scans: 
                temp, index = np.unique(np.round(self.thetaDict['{}'.format(h)], 0), return_index = True)
                xmcd = self.projMeas['{}'.format(h)][...,index]
                if temp[-1] == 360: 
                    temp = np.roll(temp, 1)
                    temp[0] = 0
                here = np.where(temp == angle)[0] + self.Params[h]['thetaoffset']
                if here >= xmcd.shape[2]: 
                    here = here-xmcd.shape[2]
                bar = ax[i, j].imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                #if i == 0: 
                #    ax[i, j].set_title('{} K'.format(h)) 

                j += 1
            i += 1
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show()
        
        if fileName != None:  
            saveName = fileName + dateToSave(time = True)
            for angle in angles2show: 
                j = 0
                for h in scans: 
                    temp, index = np.unique(np.round(self.thetaDict['{}'.format(h)], 0), return_index = True)
                    xmcd = self.projMeas['{}'.format(h)][...,index]
                    if temp[-1] == 360: 
                        temp = np.roll(temp, 1)
                        temp[0] = 0
                    here = np.where(temp == angle)[0] + self.Params[h]['thetaoffset']
                    if here >= xmcd.shape[2]: 
                        here = here-xmcd.shape[2]
                    fig, ax = plt.subplots()
                    bar = ax.imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    he = os.getcwd()
                    os.chdir(savePath)
                    plt.savefig('{}_{}K_{}Degrees.svg'.format(saveName, h, angle), dpi=1200)
                    os.chdir(he)
            plt.close('all')
            he = os.getcwd()
            os.chdir(savePath)
            fig, ax = plt.subplots()
            bar = ax.imshow(xmcd[...,here], cmap = cmap, vmin = vmin, vmax = vmax)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(bar, cax=cbar_ax)
            plt.savefig('{}_{}K_{}Degrees_bar.svg'.format(saveName, h, angle), dpi=1200)
            os.chdir(he)
    def QuiverPlot(self, direction, sliceNo, saveName = None, savePath = None): 
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
            for t in list(self.zoomedFinal.keys()):
                comps = {}
                c = (2,0)
                for i in c: 
                    comps.update({'{}'.format(i): np.swapaxes(self.zoomedFinal['{}'.format(t)][i, sliceNo], 0,1)})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)

            scale = 0.3*200/shape[:,0]
        elif direction == 'y': 
            xinterval = 1
            yinterval = 5
            shape = []
            for t in list(self.zoomedDict.keys()): 
                comps = {}
                c = (1,2)
                for i in c: 
                    comps.update({'{}'.format(i): np.swapaxes(self.zoomedFinal['{}'.format(t)][i, :,sliceNo,:], 0,1)})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)
            scale = 0.3*200/shape[:,0]
        elif direction == 'z':
            xinterval = 6
            yinterval = 6
            shape = []
            for t in list(self.zoomedDict.keys()):
                comps = {}
                c = (0,1)
                for i in c: 
                        comps.update({'{}'.format(i): self.zoomedFinal['{}'.format(t)][i, ..., sliceNo]})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(1)
            shape = np.array(shape)
            scale = shape


        fig, ax = plt.subplots(2,3, figsize = (12,6), sharex = 'all')

         # Replace with plt.savefig if you want to save a file
        i = 0
        for h in self.heatScans: 
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0 
            scale2 = 2e-1*np.nanmean(abs(mx[np.nonzero(mx)]))
            #if h == heatScans[0]: 
            #    scale2 = 2*np.mean(abs(mx))
            #else: 
            #    scale2 = 0.2*np.mean(abs(mx))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            ax[0,i].imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax[0,i].quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, scale = scale2, color = 'w', 
                          scale_units='dots') 
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_aspect(scale[i])
            i += 1
        j = 0
        for h in self.coolScans:
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0

            scale2 = 50*np.nanmean(abs(mx[np.nonzero(mx)]))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            ax[1,j].imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax[1,j].quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, color = 'w', scale = scale2,
                          )
            ax[1,j].set_xticks([])
            ax[1,j].set_yticks([])
            ax[1,j].set_aspect(scale[i+j])
            j += 1
        display_axes = fig.add_axes([0.05,0.05,0.05,0.05], projection='polar')
        display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!

        norm = mpl.colors.Normalize(-np.pi, np.pi)
        if direction == 'z':
            ll, bb, ww, hh = ax[0,0].get_position().bounds
            display_axes.set_position([1.7*ll, bb, 0.2*ww, 0.2*hh])
        else: 
            ll, bb, ww, hh = ax[0,0].get_position().bounds
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
    
    def QuiverNorm(self, direction, sliceNo, saveName = None, savePath = None): 
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        cmap = 'twilight_shifted'
        """Component = -1 gives the masks"""
        self.zoomedNorms = {temp: np.sqrt(np.nansum(self.zoomedFinal[temp]**2, axis = 0)) for temp in self.zoomedFinal.keys()}
        arr = {}
        quiverComps = {}
        if direction == 'x':
            xinterval = 1
            yinterval = 5
            shape = []
            for t in list(self.zoomedFinal.keys()):
                comps = {}
                c = (2,0)
                for i in c: 
                    temp = self.zoomedFinal['{}'.format(t)][i, sliceNo]/self.zoomedNorms['{}'.format(t)][i, sliceNo]
                    comps.update({'{}'.format(i): np.swapaxes(temp, 0,1)})
                    del temp
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)

            scale = 0.3*200/shape[:,0]
        elif direction == 'y': 
            xinterval = 1
            yinterval = 5
            shape = []
            for t in list(self.zoomedDict.keys()): 
                comps = {}
                c = (1,2)
                for i in c: 
                    temp = self.zoomedFinal['{}'.format(t)][i, :,sliceNo,:]/self.zoomedNorms['{}'.format(t)][i, :,sliceNo,:]
                    comps.update({'{}'.format(i): np.swapaxes(temp, 0,1)})
                    del temp
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)
            scale = 0.3*200/shape[:,0]
        elif direction == 'z':
            xinterval = 6
            yinterval = 6
            shape = []
            for t in list(self.zoomedDict.keys()):
                comps = {}
                c = (0,1)
                for i in c: 
                        comps.update({'{}'.format(i): self.zoomedFinal['{}'.format(t)][i, ..., sliceNo]/self.zoomedNorms['{}'.format(t)][i, ..., sliceNo]})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(1)
            shape = np.array(shape)
            scale = shape


        fig, ax = plt.subplots(2,3, figsize = (12,6), sharex = 'all')

         # Replace with plt.savefig if you want to save a file
        i = 0
        for h in self.heatScans: 
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0 
            scale2 = 2e-1*np.nanmean(abs(mx[np.nonzero(mx)]))
            #if h == heatScans[0]: 
            #    scale2 = 2*np.mean(abs(mx))
            #else: 
            #    scale2 = 0.2*np.mean(abs(mx))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            ax[0,i].imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax[0,i].quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, scale = scale2, color = 'w', 
                          scale_units='dots') 
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_aspect(scale[i])
            i += 1
        j = 0
        for h in self.coolScans:
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0

            scale2 = 50*np.nanmean(abs(mx[np.nonzero(mx)]))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            ax[1,j].imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax[1,j].quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, color = 'w', scale = scale2,
                          )
            ax[1,j].set_xticks([])
            ax[1,j].set_yticks([])
            ax[1,j].set_aspect(scale[i+j])
            j += 1
        display_axes = fig.add_axes([0.05,0.05,0.05,0.05], projection='polar')
        display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!

        norm = mpl.colors.Normalize(-np.pi, np.pi)
        if direction == 'z':
            ll, bb, ww, hh = ax[0,0].get_position().bounds
            display_axes.set_position([1.7*ll, bb, 0.2*ww, 0.2*hh])
        else: 
            ll, bb, ww, hh = ax[0,0].get_position().bounds
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
    
    def QuiverIndividual(self, direction, sliceNo, saveName = None, savePath = None): 
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
            for t in list(self.zoomedFinal.keys()):
                comps = {}
                c = (2,0)
                for i in c: 
                    comps.update({'{}'.format(i): np.swapaxes(self.zoomedFinal['{}'.format(t)][i, sliceNo], 0,1)})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)

            scale = 0.3*200/shape[:,0]
        elif direction == 'y': 
            xinterval = 1
            yinterval = 5
            shape = []
            for t in list(self.zoomedDict.keys()): 
                comps = {}
                c = (1,2)
                for i in c: 
                    comps.update({'{}'.format(i): np.swapaxes(self.zoomedFinal['{}'.format(t)][i, :,sliceNo,:], 0,1)})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(comps['{}'.format(c[0])].shape)
            shape = np.array(shape)
            scale = 0.3*200/shape[:,0]
        elif direction == 'z':
            xinterval = 6
            yinterval = 6
            shape = []
            for t in list(self.zoomedDict.keys()):
                comps = {}
                c = (0,1)
                for i in c: 
                        comps.update({'{}'.format(i): self.zoomedFinal['{}'.format(t)][i, ..., sliceNo]})
                quiverComps.update({'{}'.format(t): comps})
                shape.append(1)
            shape = np.array(shape)
            scale = shape


        # Replace with plt.savefig if you want to save a file
        i = 0
        for h in self.heatScans: 
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0 
            scale2 = 2e-1*np.nanmean(abs(mx[np.nonzero(mx)]))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            fig,ax = plt.subplots()
            ax.imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax.quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, scale = scale2, color = 'w', 
                          scale_units='dots') 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(scale[i])
            if i == 0: 
                display_axes = fig.add_axes([0.05,0.05,0.05,0.05], projection='polar')
                display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                              ## multiply the values such that 1 become 2*pi
                                              ## this field is supposed to take values 1 or -1 only!!

                norm = mpl.colors.Normalize(-np.pi, np.pi)
                if direction == 'z':
                    ll, bb, ww, hh = ax.get_position().bounds
                    display_axes.set_position([3*ll, 1.05*bb, 0.2*ww, 0.2*hh])
                else: 
                    ll, bb, ww, hh = ax.get_position().bounds
                    display_axes.set_position([ll*5, 1.05*bb, 0.5*ww, 0.5*hh])
                quant_steps = 2056
                cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap(cmap,quant_steps),
                                               norm=norm,
                                               orientation='horizontal')

                cb.outline.set_visible(False)                                 
                display_axes.set_axis_off()
            i += 1
            if saveName != None:
                saveName = saveName + dateToSave(time = True)
                here = os.getcwd()
                os.chdir(savePath)
                fig.savefig('{}_{}.svg'.format(saveName, h), dpi=1200)
                os.chdir(here)
        j = 0
        for h in self.coolScans:
            quiverKeys = list(quiverComps['{}'.format(h)].keys())
            mx = -quiverComps['{}'.format(h)]['{}'.format(quiverKeys[0])]
            my = quiverComps['{}'.format(h)]['{}'.format(quiverKeys[1])]
            m = abs(mx) > 0

            scale2 = 50*np.nanmean(abs(mx[np.nonzero(mx)]))
            x,y = np.meshgrid(np.arange(mx.shape[1]),
                              np.arange(mx.shape[0]))
            c = np.arctan2(my,mx)*m
            x[~m] = 0
            y[~m] = 0
            fig, ax = plt.subplots()
            ax.imshow(c, cmap = cmap, vmin = -np.pi, vmax = np.pi)
            ax.quiver(x[::xinterval, ::yinterval],y[::xinterval, ::yinterval],
                           mx[::xinterval, ::yinterval], my[::xinterval, ::yinterval],
                           width = 0.005, color = 'w', scale = scale2,
                          )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(scale[i+j])
            j += 1
            if saveName != None:
                saveName = saveName + dateToSave(time = True)
                here = os.getcwd()
                os.chdir(savePath)
                fig.savefig('{}_{}.svg'.format(saveName, h), dpi=1200)
                os.chdir(here)
                
    def slicePlotter(self, direction, component, sliceNo):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        """Component = -1 gives the masks"""
        if component != -1:
            arr = {}
            cmap = 'twilight_shifted'
            vmin = -1
            vmax = 1
            if direction == 'x': 
                for t in list(self.zoomedFinal.keys()): 
                    arr.update({'{}'.format(t): np.swapaxes(self.zoomedFinal['{}'.format(t)][component, sliceNo], 0,1)})
                shape = []
                for t in list(arr.keys()): 
                    shape.append(arr['{}'.format(t)].shape)
                shape = np.array(shape)
                scale = 0.3*200/max(shape[:,0])
            elif direction == 'y': 
                for t in list(self.zoomedFinal.keys()): 
                    arr.update({'{}'.format(t): np.swapaxes(self.zoomedFinal['{}'.format(t)][component, :, sliceNo, :],0,1)})
                shape = []
                for t in list(arr.keys()): 
                    shape.append(arr['{}'.format(t)].shape)
                shape = np.array(shape)
                scale = 0.3*200/max(shape[:,0])
            elif direction == 'z': 
                for t in list(self.zoomedFinal.keys()): 
                    arr.update({'{}'.format(t): self.zoomedFinal['{}'.format(t)][component, :, :, sliceNo]})
                scale = 1
        else: 
            arr = {}
            cmap = 'Greys'
            vmin = -1
            vmax = 1
            if direction == 'x': 
                for t in list(self.zoomedMasks.keys()): 
                    arr.update({'{}'.format(t): np.swapaxes((self.zoomedMasks['{}'.format(t)][sliceNo]), 0,1)})
                shape = []
                for t in list(arr.keys()): 
                    shape.append(arr['{}'.format(t)].shape)
                shape = np.array(shape)
                scale = 0.3*200/max(shape[:,0])
            elif direction == 'y': 
                for t in list(self.magDomains.keys()): 
                    arr.update({'{}'.format(t): np.swapaxes((self.zoomedMasks['{}'.format(t)][:, sliceNo, :]),0,1)})
                shape = []
                for t in list(arr.keys()): 
                    shape.append(arr['{}'.format(t)].shape)
                shape = np.array(shape)
                scale = 0.3*200/max(shape[:,0])
            elif direction == 'z': 
                for t in list(self.magDomains.keys()): 
                    arr.update({'{}'.format(t): (self.zoomedMasks['{}'.format(t)][...,sliceNo])})
                scale = 1



        fig, ax = plt.subplots(2,3, figsize = (12,6), sharey = 'all', sharex = 'all')
        i = 0
        for h in heatScans:  
            ax[0,i].imshow(arr['{}'.format(h)], vmin = vmin, vmax = vmax, cmap = cmap)
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_aspect(scale)
            i += 1
        j = 0
        for c in coolScans:  
            ax[1,j].imshow(arr['{}'.format(c)], vmin = vmin, vmax = vmax, cmap = cmap)
            ax[1,j].set_xticks([])
            ax[1,j].set_yticks([])
            ax[1,j].set_aspect(scale)
            j += 1
        plt.tight_layout()