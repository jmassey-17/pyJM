# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:42:56 2022

@author: massey_j
"""

"""Ideally we could have this do the rotation, apply it to the stack"""
"""Then the cropping"""
"""Then the thresholding against the magnetization magnitude"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import json
import scipy
import time
import threading

class RotationalGUI:
    
    def __init__(self, im3D, **kwargs):
        
        image = np.sum(im3D, axis = 2)
        self.image = image
        self.flag = 0
        self._default_params(**kwargs)
        self.gui()
        
    def _default_slider_dict(self, label, valmin, valinit, valmax, valstep):
        
        return {"label" : label, "valmin" : valmin, "valinit" : valinit, "valmax" : valmax, "valstep" : valstep}
        
    def _default_params(self, **kwargs):
    
        self.axis_color = kwargs.pop("axis_color", "lightgoldenrodyellow")

        self.save_dir = kwargs.pop("save_dir", None)
        
        self.theta_valinit = kwargs.pop("theta",0.0)
        self.theta_valmin = kwargs.pop("thetamin",-90)
        self.theta_valmax = kwargs.pop("thetamax",90)
        self.theta_valstep = kwargs.pop("thetastep",1)
        self.theta_label = "$\\theta$: "

        self.theta_defaults = self._default_slider_dict(self.theta_label, self.theta_valmin, self.theta_valinit, self.theta_valmax, self.theta_valstep)
        
    def _create_axes(self):

        self.theta_forward_ax  = self.fig.add_axes([0.945, 0.25, 0.03, 0.06], facecolor=self.axis_color)
        self.theta_backward_ax  = self.fig.add_axes([0.91, 0.25, 0.03, 0.06], facecolor=self.axis_color)

        self.theta_slider_ax  = self.fig.add_axes([0.525, 0.25, 0.325, 0.06], facecolor=self.axis_color)
        
        self.slider_axes = [self.theta_slider_ax]

    def _init_buttons(self):

        self.theta_f_button = Button(self.theta_forward_ax, u"\u25B6", color='w', hovercolor='lightgrey')
        self.theta_b_button = Button(self.theta_backward_ax, u'\u25C0', color='w', hovercolor='lightgrey')  

    def _init_sliders(self):   
    
        self.theta_slider = Slider(self.theta_slider_ax,self.theta_label, valmin = self.theta_valmin, valmax = self.theta_valmax, valinit=self.theta_valinit, valstep = self.theta_valstep)
    
        self.sliders = [self.theta_slider]
        
        self.nSliders = len(self.sliders)
       
    def sliders_on_changed(self, val):

        rotated = scipy.ndimage.rotate(self.image, self.theta_slider.val, reshape = False)

        self.ax[1].imshow(rotated)
    
    def save(self, val):
        
        for i in range(self.nSliders):
            
            self.sliders[i].valinit = self.sliders[i].val
            self.sliders[i].vline.set_data([self.sliders[i].val, self.sliders[i].val], [0, 1])

        self.fig.canvas.draw_idle()
        
    def _set_slider(self, slider, params):

        slider.valmin = params["valmin"]
        slider.valinit = params["valinit"]
        slider.valmax = params["valmax"]
        slider.label = params["label"]
        slider.vline.set_data([params["valinit"], params["valinit"]], [0, 1])
        slider.reset()
        
    def reset(self, val):

        for slider in self.sliders:
            slider.reset()
    
    def happy(self):
        self.flag = 1
        plt.close(self.fig.number)

    
    def forward(self,val, slider):
        
        init_value = slider.valinit
        slider.valinit = slider.val + slider.valstep
        slider.reset()
        slider.valinit = init_value
        
    def backward(self,val, slider):
        
        init_value = slider.valinit
        slider.valinit = slider.val - slider.valstep
        slider.reset()
        slider.valinit = init_value

    def _save(self):
        
        if self.save_dir != None:

            with open(self.save_dir + 'params.json', 'w') as fp:
                json.dump(self.params, fp)

            np.savez(self.save_dir + 'points', self.points)
            np.savez(self.save_dir + 'indices', self.indices)
            np.savez(self.save_dir + 'mask', self.mask)

    def _generate_mask(self, verbose = True):

        # Number of rows in the best fit lattice
        nElements = self.points.shape[0]

        self.mask = np.nan * np.zeros((self.nRows, self.nCols))

        # Now iterate over all the elements in the lattice
        for e in range(0, nElements):
            exact_centre = self.points[e,:]
            diameter = np.ceil(self.params["d"])

            # Need integer values for the purposes of rounding
            x = int(round(exact_centre[0]))
            y = int(round(exact_centre[1]))
            d = int(diameter)

            xstart = x - d if (x-d)>=0 else 0
            xstop = x + d if (x+d)<=self.nRows else self.nRows
            ystart = y - d if (y-d)>0 else 0
            ystop = y + d if (y+d)<=self.nCols else self.nCols

            for i in range(xstart, xstop):
                for j in range(ystart, ystop):

                    dist = dist = np.sqrt((i-exact_centre[0])**2 + (j-exact_centre[1])**2)
                    if dist <= diameter/2:
                        self.mask[j,i] = e


        unique, counts = np.unique(self.mask, return_counts = True)

        if verbose:
            print("")
            print("Mask values")
            print("===========")
            print("nElements:", nElements)
            print("Px. per element:", round(np.ma.mean(np.ma.masked_array(counts, mask = np.isnan(unique))),ndigits=2), "+/-", round(np.ma.masked_array.std(np.ma.masked_array(counts, mask = np.isnan(unique))),ndigits=2))
            print("Max px.:\t", np.ma.maximum.reduce(np.ma.masked_array(counts, mask = np.isnan(unique))).data)
            print("Min px.:\t", np.ma.minimum.reduce(np.ma.masked_array(counts, mask = np.isnan(unique))).data)
        
    def default(self, val):
        
        self._set_slider(self.theta_slider, self.theta_defaults)



    def gui(self):
        
        #plt.close("all")
        self.fig, self.ax = plt.subplots(1,2, figsize=(8,4))
        self.rotated = scipy.ndimage.rotate(self.image, 0, reshape = False)
        self.ax[0].imshow(self.image)
        self.ax[1].imshow(self.rotated)

        self.fig.subplots_adjust(left=0.4, bottom=0.4)

        self._create_axes()
        self._init_sliders()
        self._init_buttons()
        
        self.default_box = self.fig.add_axes([0.525, 0.15, 0.1, 0.06])
        self.default_button = Button(self.default_box, "Default")
        
        self.save_box = self.fig.add_axes([0.641, 0.15, 0.1, 0.06])
        self.save_button = Button(self.save_box, "Save")
        
        self.reset_box = self.fig.add_axes([0.757, 0.15, 0.1, 0.06])
        self.reset_button = Button(self.reset_box, "Reset")
        
        self.happy_box = self.fig.add_axes([0.873, 0.15, 0.1, 0.06])
        self.happy_button = Button(self.happy_box, "Happy")

        
        self.default_button.on_clicked(self.default)
        self.reset_button.on_clicked(self.reset)
        self.save_button.on_clicked(self.save)
        self.happy_button.on_clicked(self.happy)
        
        self.theta_slider.on_changed(self.sliders_on_changed)

        self.theta_f_button.on_clicked(lambda val: self.forward(val, self.theta_slider))
        self.theta_b_button.on_clicked(lambda val: self.backward(val, self.theta_slider))
        
        self.fig.show()




class ThresholdGUI:
    
    def __init__(self, image3D, xmcdImages, theta, thetaOffset, **kwargs):
        self.theta = theta
        self.flag = 0
        self.thetaOffset = thetaOffset
        self.xmcdImages = xmcdImages
        self.image = np.sum(np.sqrt(image3D[0]**2 + image3D[1]**2 + image3D[2]**2), axis = 2)
        self._default_params(**kwargs)
        
        self.gui()
        
    def _default_slider_dict(self, label, valmin, valinit, valmax, valstep):
        
        return {"label" : label, "valmin" : valmin, "valinit" : valinit, "valmax" : valmax, "valstep" : valstep}
        
    def _default_params(self, **kwargs):
    
        self.axis_color = kwargs.pop("axis_color", "lightgoldenrodyellow")

        self.save_dir = kwargs.pop("save_dir", None)
        
        self.theta_valinit = kwargs.pop("theta",0.0)
        self.theta_valmin = kwargs.pop("thetamin",0)
        self.theta_valmax = kwargs.pop("thetamax",1)
        self.theta_valstep = kwargs.pop("thetastep",0.01)
        self.theta_label = "$\\theta$: "

        self.theta_defaults = self._default_slider_dict(self.theta_label, self.theta_valmin, self.theta_valinit, self.theta_valmax, self.theta_valstep)
        
    def _create_axes(self):

        self.theta_forward_ax  = self.fig.add_axes([0.945, 0.25, 0.03, 0.06], facecolor=self.axis_color)
        self.theta_backward_ax  = self.fig.add_axes([0.91, 0.25, 0.03, 0.06], facecolor=self.axis_color)

        self.theta_slider_ax  = self.fig.add_axes([0.525, 0.25, 0.325, 0.06], facecolor=self.axis_color)
        
        self.slider_axes = [self.theta_slider_ax]

    def _init_buttons(self):

        self.theta_f_button = Button(self.theta_forward_ax, u"\u25B6", color='w', hovercolor='lightgrey')
        self.theta_b_button = Button(self.theta_backward_ax, u'\u25C0', color='w', hovercolor='lightgrey')  

    def _init_sliders(self):   
    
        self.theta_slider = Slider(self.theta_slider_ax,self.theta_label, valmin = self.theta_valmin, valmax = self.theta_valmax, valinit=self.theta_valinit, valstep = self.theta_valstep)
    
        self.sliders = [self.theta_slider]
        
        self.nSliders = len(self.sliders)
       
    def sliders_on_changed(self, val):

        threshedImage = np.copy(self.image, order = "C")
        m = threshedImage > self.theta_slider.val*np.amax(threshedImage)
        mag2show = np.copy(threshedImage)
        mag2show[~m] = 0
        

        self.ax[0,1].imshow(mag2show)
    
    def save(self, val):
        
        for i in range(self.nSliders):
            
            self.sliders[i].valinit = self.sliders[i].val
            self.sliders[i].vline.set_data([self.sliders[i].val, self.sliders[i].val], [0, 1])

        self.fig.canvas.draw_idle()
        
    def _set_slider(self, slider, params):

        slider.valmin = params["valmin"]
        slider.valinit = params["valinit"]
        slider.valmax = params["valmax"]
        slider.label = params["label"]
        slider.vline.set_data([params["valinit"], params["valinit"]], [0, 1])
        slider.reset()
        
    def reset(self):

        for slider in self.sliders:
            slider.reset()
    
    def happy(self):
        self.flag = 1
        plt.close(self.fig.number)


    def forward(self,val, slider):
        
        init_value = slider.valinit
        slider.valinit = slider.val + slider.valstep
        slider.reset()
        slider.valinit = init_value
        
    def backward(self,val, slider):
        
        init_value = slider.valinit
        slider.valinit = slider.val - slider.valstep
        slider.reset()
        slider.valinit = init_value

    def _save(self):
        
        if self.save_dir != None:

            with open(self.save_dir + 'params.json', 'w') as fp:
                json.dump(self.params, fp)

            np.savez(self.save_dir + 'points', self.points)
            np.savez(self.save_dir + 'indices', self.indices)
            np.savez(self.save_dir + 'mask', self.mask)

    def _generate_mask(self, verbose = True):

        # Number of rows in the best fit lattice
        nElements = self.points.shape[0]

        self.mask = np.nan * np.zeros((self.nRows, self.nCols))

        # Now iterate over all the elements in the lattice
        for e in range(0, nElements):
            exact_centre = self.points[e,:]
            diameter = np.ceil(self.params["d"])

            # Need integer values for the purposes of rounding
            x = int(round(exact_centre[0]))
            y = int(round(exact_centre[1]))
            d = int(diameter)

            xstart = x - d if (x-d)>=0 else 0
            xstop = x + d if (x+d)<=self.nRows else self.nRows
            ystart = y - d if (y-d)>0 else 0
            ystop = y + d if (y+d)<=self.nCols else self.nCols

            for i in range(xstart, xstop):
                for j in range(ystart, ystop):

                    dist = dist = np.sqrt((i-exact_centre[0])**2 + (j-exact_centre[1])**2)
                    if dist <= diameter/2:
                        self.mask[j,i] = e


        unique, counts = np.unique(self.mask, return_counts = True)

        if verbose:
            print("")
            print("Mask values")
            print("===========")
            print("nElements:", nElements)
            print("Px. per element:", round(np.ma.mean(np.ma.masked_array(counts, mask = np.isnan(unique))),ndigits=2), "+/-", round(np.ma.masked_array.std(np.ma.masked_array(counts, mask = np.isnan(unique))),ndigits=2))
            print("Max px.:\t", np.ma.maximum.reduce(np.ma.masked_array(counts, mask = np.isnan(unique))).data)
            print("Min px.:\t", np.ma.minimum.reduce(np.ma.masked_array(counts, mask = np.isnan(unique))).data)
        
    def default(self, val):
        
        self._set_slider(self.theta_slider, self.theta_defaults)

    def show_mask(self):

        plt.close("all")
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        fig.canvas.manager.set_window_title('Mask comparison')

        axs[0].imshow(self.image)
        axs[0].set_title("Image")

        patch_collection = []
        
        for i in range(self.points.shape[0]):
            patch_collection.append(Circle((self.points[i,0],self.points[i,1]), self.params["d"]/2))
        
        p = PatchCollection(patch_collection, facecolor="none", edgecolor="white")
        axs[0].add_collection(p)

        axs[1].imshow(self.mask)
        axs[1].set_title("Mask")

        fig.show()

    def gui(self):

        plt.close("all")
        self.fig, self.ax = plt.subplots(2,2, figsize=(8,4))
        self.ax[0,0].imshow(self.image)
        self.ax[0,1].imshow(self.image)
        
        temp, index = np.unique(np.round(self.theta, 0), return_index = True)
        if temp[-1] == 360: 
            temp = np.roll(temp, 1)
            temp[0] = 0
        angles2show = [0, 94]
        j = 0
        for angle in angles2show: 
            here = np.where(temp == angle)[0] + self.thetaOffset
            if here >= self.xmcdImages.shape[2]: 
                here = here-self.xmcdImages.shape[2]
            self.ax[1, j].imshow(self.xmcdImages[...,here], vmin = -0.05, vmax = 0.05)
            j += 1


        self.fig.subplots_adjust(left=0.4, bottom=0.4)

        self._create_axes()
        self._init_sliders()
        self._init_buttons()
        
        self.default_box = self.fig.add_axes([0.525, 0.15, 0.1, 0.06])
        self.default_button = Button(self.default_box, "Default")
        
        self.save_box = self.fig.add_axes([0.641, 0.15, 0.1, 0.06])
        self.save_button = Button(self.save_box, "Save")
        
        self.reset_box = self.fig.add_axes([0.757, 0.15, 0.1, 0.06])
        self.reset_button = Button(self.reset_box, "Reset")
        
        self.happy_box = self.fig.add_axes([0.873, 0.15, 0.1, 0.06])
        self.happy_button = Button(self.happy_box, "Happy")

        
        self.default_button.on_clicked(self.default)
        self.reset_button.on_clicked(self.reset)
        self.save_button.on_clicked(self.save)
        self.happy_button.on_clicked(self.happy)
        
        self.theta_slider.on_changed(self.sliders_on_changed)

        self.theta_f_button.on_clicked(lambda val: self.forward(val, self.theta_slider))
        self.theta_b_button.on_clicked(lambda val: self.backward(val, self.theta_slider))
        
        self.fig.show()



class CropGUI:
    
    def __init__(self, image3D, **kwargs):
        self.flag = 0
        self.imageStack = image3D
        self.image = np.sum(np.sqrt(image3D[0]**2 + image3D[1]**2 + image3D[2]**2), axis = 2)
    
        self.corners = self.get_raw_corners()
        
        
    def double_click_and_mark(self, event, axs, container, nPoints):
        
        if event.dblclick:
            
            axs.plot(event.xdata, event.ydata, marker="x", color="r",zorder=3)
            axs.figure.canvas.draw()
            
            container.append([event.xdata, event.ydata])
            
            if len(container)== nPoints:
                plt.close("all")

    def get_raw_corners(self):

        corners = []
        nCorners = 4

        plt.close("all")
        fig, axs  = plt.subplots() 
        fig.canvas.manager.set_window_title("Choose ROI")
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.double_click_and_mark(event, axs, corners, nCorners)) 
        axs.imshow(self.image)
        
        plt.show()
        count = 0
        next = 0.
        while plt.fignum_exists(fig.number):

            while time.time() < next:
                plt.pause(.001) 

            next = time.time() + 1.0  
            count += 1

        return np.array(corners)
    
        
    def gui(self):

        corners = []
        nCorners = 4

        #plt.close("all")
        fig, axs  = plt.subplots() 
        fig.canvas.manager.set_window_title("Choose ROI")
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.double_click_and_mark(event, axs, corners, nCorners)) 
        axs.imshow(self.image)
        
        plt.show()
        count = 0
        next = 0.
        while plt.fignum_exists(fig.number):

            while time.time() < next:
                plt.pause(.001) 

            next = time.time() + 1.0  
            count += 1
        if count == 3: 
            self.flag = 1
        return np.array(corners)

