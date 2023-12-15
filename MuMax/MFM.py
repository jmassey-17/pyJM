# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 08:43:34 2023

@author: massey_j-adm

05/12/23
Edits by Sam:
    Labels on graphs. Images now plotted with the correct orientation and scale
    Added decode(encoding) in loadAndProcessChannels remove if this is an error later on
    Image alignment code works for individual images
        Still tidying the code to work for multiple files
        
06/12/23
    Alignment code now works for files within class
        Still need to add a padding to the shifts to prevent features going from top to bottom of the image
        
"""


#Appending path to pyMFM
import igor.binarywave as bw
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift



""" Load the Igor wave """
class MFMFile():
    """
    class to load and process Oxford MFM data
    
    """
    def __init__(self, filename): 
        """
        initializes the MFMFile object

        Parameters
        ----------
        filename : str
            filename to load.

        Returns
        -------
        None.

        """
        self.wave = bw.load(filename)
        self.data = self.wave['wave']['wData']
        self.loadAndProcessChannels()
        self.generateMetadata()
    
    def loadAndProcessChannels(self):
        """
        reads channels contained in the file and loads them to the object

        Returns
        -------
        None.

        """
        encoding = "latin1"
        labels = self.wave['wave']['labels'][2]
        channel_labels = [labels[channel_number].decode(encoding) for channel_number in range(1, len(labels))] #decode(encoding) to allow for titles to be the proper titles without letter b
        self.channels = channel_labels
        for i,l in enumerate(channel_labels): 
            l = str(l).replace('b', '')
            l = str(l).replace('\'', '')
            setattr(self, l, self.data[...,i])
    
    def generateMetadata(self): 
        """
        generates metadata for image

        Returns
        -------
        None.

        """
        self.metadata = {}
        encoding = 'latin1'
        for t in self.wave['wave']['note'].decode(encoding).split('\r'): 
            try: 
                self.metadata.update({t.split(':')[0].replace(':', '') : t.split(':')[1].replace(' ', '')})
            except: 
                pass
            
    def planeLevel(self, show_plot = False):
        """
        fits and removes plane from all channels
        
        A simple routine to subtract a best fit plane from the image.
        
        Based on: https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array?fbclid=IwAR2jQtc45L87z4PQ-b-6y4VlK71PnXE45j_KvqAp8g-BTtj_RFMacfjlIXM

        Parameters
        ----------
        show_plot : Bool, optional
            . The default is False.

        Returns
        -------
        None.

        """
             
       
        for channel in self.channels:
            array = (getattr(self, channel))
            
            X1, X2 = np.mgrid[:array.shape[0], :array.shape[1]]
            
            X = np.hstack(( np.reshape(X1, (array.shape[0]*array.shape[0], 1)) , np.reshape(X2,(array.shape[1]*array.shape[1], 1)) ) )
            X = np.hstack(( np.ones((array.shape[0]*array.shape[1], 1)) , X ))
            channel_data_reshaped = np.reshape(array, (array.shape[0]*array.shape[1], 1))
            
            theta_data = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), channel_data_reshaped)
            
            background = np.reshape(np.dot(X, theta_data), (array.shape[0], array.shape[1]))
            levelled = array - background
            levelled_rotated = np.swapaxes(levelled, 0, 1)
            
            # Do the subtraction
            setattr(self, f'{channel}_flattened', levelled_rotated)
            test = 0
            try: 
                test = getattr(self, 'flattenedChannels')
            except: 
                pass
            
            if test == 0: 
                self.flattenedChannels = [f'{channel}_flattened']
            else: 
                self.flattenedChannels.append(f'{channel}_flattened')
                      
            
            
        if (show_plot) & (channel == 'NapPhaseRetrace'):
           fig, axs = plt.subplots(1,3, figsize=(12,4))
           
           axs[0].imshow(array,origin='lower',cmap="inferno")
           axs[1].imshow(levelled,origin='lower',cmap="inferno", vmin=-0.4, vmax=0.4) #Note this plot is different to real data, but keeps consistant with the other plots until now
           axs[2].imshow(background,origin='lower',cmap="inferno")
       
           axs[0].set_title("Raw")
           axs[1].set_title("Levelled")
           axs[2].set_title("Background")
             
           fig.tight_layout()
    
    
    
    def testAggregateFunction(self, attributeToTest, aggregateFunction, axis = 1, show_plot = False):
        """
        method to test function and axis by which to make the best binary images for feature extraction

        Parameters
        ----------
        attributeToTest : str
            channel information to test.
        aggregateFunction : func
            function to test.
        axis : int, optional
            axis over which to test the aggregate function. The default is 1.
        show_plot : Bool, optional
            . The default is False.

        Raises
        ------
        ValueError
            ('Axis must be either 0 (x) or 1 (y)').

        Returns
        -------
        None.

        """
        array = getattr(self, attributeToTest)
        pos, neg = np.zeros_like(array), np.zeros_like(array)
        if axis == 0 or axis == 'x': 
            for i in range(array.shape[0]): 
                pos[i] = array[i] > aggregateFunction(array[i])
                neg[i] = array[i] < aggregateFunction(array[i])
        elif axis == 1 or axis == 'y': 
            for i in range(array.shape[1]): 
                pos[:,i] = array[:,i] > aggregateFunction(array[:,i])
                neg[:,i] = array[:,i] < aggregateFunction(array[:,i])
        else: 
            raise ValueError('Axis must be either 0 (x) or 1 (y)')
            
        if show_plot:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(array,origin='lower',cmap="inferno", vmin=-0.4, vmax=0.4)
            ax[1].imshow(pos,origin='lower',cmap="inferno")
            ax[2].imshow(neg,origin='lower',cmap="inferno")
            
            ax[0].set_title("Flattened")
            ax[1].set_title("Positive")
            ax[2].set_title("Negative")
        
        
    def domainAnalysis(self, attributeToTest, aggregateFunction, axis = 0): 
        """
        runs a regionprops for up and down and calcualtes skyrmion winding number for each area

        Returns
        -------
        None.

        """
        from skimage.measure import label,  regionprops_table
        array = getattr(self, attributeToTest)
        pos, neg = np.zeros_like(array), np.zeros_like(array)
        if axis == 0 or axis == 'x': 
            for i in range(array.shape[0]): 
                pos[i] = array[i] > aggregateFunction(array[i])
                neg[i] = array[i] < aggregateFunction(array[i])
        elif axis == 1 or axis == 'y': 
            for i in range(array.shape[1]): 
                pos[:,i] = array[:,i] > aggregateFunction(array[:,i])
                neg[:,i] = array[:,i] < aggregateFunction(array[:,i])
        else: 
            raise ValueError('Axis must be either 0 (x) or 1 (y)')
            
        self.positiveMask = pos
        self.negativeMask = neg
            
        positive = label(pos)
        negative = label(neg)
        types = ['positive', 'negative']
        
        for i, im in enumerate((positive, negative)): 
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'image_filled',
                                             'eccentricity',
                                             'axis_major_length',
                                             'axis_minor_length')))
            
            offScreen_x = ((regions['bbox-0'] == 0) | (regions['bbox-2'] == im.shape[0]))
            offScreen_y = ((regions['bbox-1'] == 0) | (regions['bbox-3'] == im.shape[1]))
            
            onScreen = regions[(~offScreen_x & ~offScreen_y)]

            if len(onScreen) > 0: 
                onScreen['type'] = types[i]
                onScreen['field'] = np.round(float(self.metadata['MagneticField']), 2)
            
            if i == 0: 
                self.domains = onScreen
            else: 
                self.domains = pd.concat([self.domains, onScreen], ignore_index = True)


    def alignImages(self, referenceImage, offsetImage, show_plot = False, pad=True):
        """
        aligns offsetImage to referenceImage
        Based on: https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html

        Parameters
        ----------
        referenceImage : np.array
        offsetImage : np.array
        show_plot : Bool, optional
             The default is False.
        pad : Bool, optional
            pad array with zeros. The default is True.

        Returns
        -------
        None.

        """
        
        shift, error, diffphase = phase_cross_correlation(referenceImage, offsetImage, upsample_factor=100)
        print(f'Detected subpixel offset (y, x): {shift}')
    
        try: 
            for channel in self.flattenedChannels:
                if pad:
                    padx = np.abs(int(shift[1])) + 1
                    pady = np.abs(int(shift[0])) + 1
                    pad_vals = ([pady]*2,[padx]*2)
                    im = np.pad(getattr(self, channel),pad_vals,'constant')
                else:
                    padx = 0; pady = 0
                    im = getattr(self, channel)

                #shiftedImage = np.fft.ifftn(fourier_shift(np.fft.fftn(getattr(self, channel)), shift)).real
                shiftedImage = np.fft.ifftn(fourier_shift(np.fft.fftn(im), shift)).real
                shiftedImage =  shiftedImage[pady:pady+ getattr(self, channel).shape[0], padx:padx+ getattr(self, channel).shape[1]]
                
                setattr(self, f'{channel}_shifted',shiftedImage)
                setattr(self, f'{channel}_shift',shift)
                
                if (show_plot) & (channel == 'AmplitudeRetrace_flattened'):
                    
                    plt.figure(figsize=(8, 5), dpi = 300)
                    ax1 = plt.subplot(1, 2, 1)
                    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
                    
                    ax1.imshow(referenceImage, cmap='gray')
                    ax1.set_axis_off()
                    ax1.set_title('Reference image',fontsize = 5)

                    ax2.imshow(shiftedImage, cmap='gray')
                    ax2.set_axis_off()
                    ax2.set_title('Offset image after shifting', fontsize = 5)
                    
                    plt.tight_layout()
                    plt.savefig(save_path + f"AFM of Field_sweep {i} shifted with pad upsample 1000.png",dpi=150)
                    plt.show()
                
        except: 
            pass
   
    
   
    
   
    def saveShiftedImages(self):
       for j in range(0,len(fileList)):
            plt.imshow(shiftedImages[...,j], cmap='inferno',vmin=-0.4, vmax=0.4)   
            plt.savefig(save_path + f"{j} shifted MFM image.png", dpi=150)
         
            
         
            
         
            
         
####### User definitions #######
wkdir = r'C:\Users\treves_s\Documents\PSI\Measurements\MFM\Oxford\20230914_FIB_pills_samp_4\FIB_Pills\Pills_field_sweep_part_1'
fileList = [file for file in os.listdir(wkdir) if file.find('.ibw') != -1]

save_path = wkdir + r'/Analysed_folder/'

if not os.path.exists(save_path):
    os.mkdir(save_path)



sampPerLine = 512 #How many samples per line there are for your scan
shiftedImages = np.zeros(shape = (sampPerLine, sampPerLine, len(fileList)))
shifts = np.zeros(shape = (len(fileList), 2))

for i, file in enumerate(fileList): 
    filename = os.path.join(wkdir, file)
    imageData = MFMFile(filename)
    field = np.round(float(imageData.metadata['MagneticField']),2)
    print(f'Processing {field}')
    imageData.planeLevel(imageData, show_plot=False)
    imageData.testAggregateFunction('NapPhaseRetrace_flattened', np.median, axis = 0, show_plot=False) #Due to image being rotated, axis analysed may be changed
    imageData.domainAnalysis('NapPhaseRetrace_flattened', np.median, axis = 0)
    
    if i == 0:
            #referenceImage = imageData.HeightRetrace
            referenceImage = imageData.AmplitudeRetrace_flattened
    
    imageData.alignImages(referenceImage, imageData.HeightRetrace_flattened, show_plot=False)
    shiftedImages[...,i] = imageData.NapPhaseRetrace_flattened_shifted
    shifts[i,...] = imageData.NapPhaseRetrace_flattened_shift
    
    
    if i == 0: 
        total = imageData.domains
    else: 
        total = pd.concat([total, imageData.domains], ignore_index=True)
        
imageData.saveShiftedImages()
    
        
 