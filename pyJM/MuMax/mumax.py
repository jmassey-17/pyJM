# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:14:28 2023

@author: massey_j-adm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('mode.chained_assignment', None) #gets rid of errors that are unclear how to handle...
import seaborn as sns
from pyJM.BasicFunctions import timeOperation


def _read_line(data, metadata):
    """
    reads line and adds to metadata
    hacked from Stoner.formats.simulations

    Parameters
    ----------
    data : file
        file to strip to metadata.
    metadata : dict
        metadata dictionary.

    Returns
    -------
    None

    """
    """Read a single line and add to a metadata dictionary."""
    line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
    if line == "" or ":" not in line:
        return True
    parts = line.split(":")
    field = parts[0].strip()
    value = ":".join(parts[1:]).strip()
    if field == "Begin" or field == 'End':
        if value.startswith("Data "):
            value = value.split(" ")
            metadata["representation"] = value[1]
            if value[1] == "Binary":
                metadata["representation size"] = value[2]
            return False
        else: 
            return True
    if field not in ["Begin", "End"]:
        metadata[field] = value
        return True


def E(i,j,k): 
    """
    returns levi-civita value for i,j,k

    Parameters
    ----------
    i : int
    j : int
    k : int.

    Returns
    -------
    levi_civita value: int

    """
    return int((i-j)*(j-k)*(k-i)/2)

def calculateWindingNumber2(image, cx, cy, image_filled, window = 5):
    """
    will calculate the winding number in defined area + window
    for use with regionprops

    Parameters
    ----------
    image : np.array
        component of magnetization.
    cx : int
        centre in x direction of area.
    cy : int
        centre in x direction of area.
    image_filled : np.array
        result from regionprops.
    window : int, optional
        size of window around defined area. The default is 5.

    Returns
    -------
    windingNumber: float
        windingNumber of the area.

    """
    s = max(image_filled.shape)
    x1 = int(cx - int(s/2)) - window
    x2 = int(cx + int(s/2)) + window
    y1 = int(cy - int(s/2)) - window
    y2 = int(cy + int(s/2)) + window
    if y1 < 0: 
        y1,y2 = 0, s

    m = image[:,x1:x2, y1:y2]
    try:
        dmdx = np.gradient(m, axis = 1)
        dmdy = np.gradient(m, axis = 2)
        crossProduct = np.cross(dmdx, dmdy, axis = 0)
        dotProduct = np.sum(crossProduct*m, axis = 0)
        return (1/(4*np.pi))*np.sum(dotProduct)
    except: 
        return np.nan


class mumax:
    """JM 20231117 
    hacked from Stoner.formats.simulations
    
    A class that reads OOMMF vector format files and constructs x,y,z,u,v,w data.

    OVF 1 and OVF 2 files with text or binary data and only files with a meshtype rectangular are supported
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.ovf"]  # Recognised filename patterns

    mime_type = ["text/plain", "application/octet-stream"]

    def __init__(self, filename, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards.

        Notes:
            This code can handle only the first segment in the data file.
        """
        self.filename = filename
        self.metadata = {}
        # Reading in binary, converting to utf-8 where we should be in the text header
        with open(self.filename, "rb") as data:
            line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
            if line not in ["OOMMF: rectangular mesh v1.0", "OOMMF OVF 2.0"]:
                raise NameError('HiThere')("Not an OVF 1.0 or 2.0 file.")
       
            while _read_line(data, self.metadata):
                pass  # Read the file until we reach the start of the data block.
           
            if self.metadata["representation"] == "Binary":
                size = (  # Work out the size of the data block to read
                    int(self.metadata["xnodes"])
                    * int(self.metadata["ynodes"])
                    * int(self.metadata["znodes"])
                    * int(self.metadata["valuedim"])
                    + 1
                ) * int(self.metadata["representation size"])
                bin_data = data.read(size)
                numbers = np.frombuffer(bin_data, dtype=f"<f{self.metadata['representation size']}")
                chk = numbers[0]
                if (
                    chk != [1234567.0, 123456789012345.0][int(self.metadata["representation size"]) // 4 - 1]
                ):  # If we have a good check number we can carry on, otherwise try the other endianess
                    numbers = np.frombuffer(bin_data, dtype=f">f{self.metadata['representation size']}")
                    chk = numbers[0]

                data = np.reshape(
                    numbers[1:], (int(self.metadata["xnodes"]) * int(self.metadata["ynodes"]) * int(self.metadata["znodes"]), 3),
                )
            else:
                data = np.genfromtxt(
                    data, max_rows=int(self.metadata["xnodes"]) * int(self.metadata["ynodes"]) * int(self.metadata["znodes"]),
                )
        xmin, xmax, xstep = (
            float(self.metadata["xmin"]),
            float(self.metadata["xmax"]),
            float(self.metadata["xstepsize"]),
        )
        ymin, ymax, ystep = (
            float(self.metadata["ymin"]),
            float(self.metadata["ymax"]),
            float(self.metadata["ystepsize"]),
        )
        zmin, zmax, zstep = (
            float(self.metadata["zmin"]),
            float(self.metadata["zmax"]),
            float(self.metadata["zstepsize"]),
        )
        Z, Y, X = np.meshgrid(
            np.arange(zmin + zstep / 2, zmax, zstep) * 1e9,
            np.arange(ymin + ystep / 2, ymax, ystep) * 1e9,
            np.arange(xmin + xstep / 2, xmax, xstep) * 1e9,
            indexing="ij",
        )
        self.data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), data))
        self.m = data.reshape((int(self.metadata['znodes']), int(self.metadata['xnodes']), int(self.metadata['ynodes']), 3))
        self.m = np.swapaxes(self.m, 0, -1)
        

    def findAndCharacterizeSkyrmions(self): 
        """
        runs a regionprops for up and down and calcualtes skyrmion winding number for each area

        Returns
        -------
        None.

        """
        from skimage.measure import label, regionprops_table
        
        image = self.m[...,0] 
        positive = label(image[2] > 0)
        negative = label(image[2] < 0)
        types = ['positive', 'negative']
        
        for i, im in enumerate((positive, negative)): 
            regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                             'area', 
                                             'bbox', 
                                             'image_filled',
                                             'eccentricity',
                                             'axis_major_length',
                                             'axis_minor_length')))
            # Remove those that go off the screen
            offScreen_x = ((regions['bbox-0'] == 0) | (regions['bbox-2'] == im.shape[0]))
            offScreen_y = ((regions['bbox-1'] == 0) | (regions['bbox-3'] == im.shape[1]))
            
            onScreen = regions[(~offScreen_x & ~offScreen_y)]
            if len(onScreen) > 0: 
                onScreen['Q'] = onScreen.apply(lambda row: calculateWindingNumber2(image, row['centroid-0'], row['centroid-1'], row['image_filled'], 5), axis = 1)
                onScreen['type'] = types[i]
            try: 
                test = self.onScreen
            except: 
                test = 0
                
            if i == 0 or test == 0: 
                self.skyrmionAnalysis = onScreen
            else: 
                self.skyrmionAnalysis = pd.concat([self.skyrmionAnalysis, onScreen], ignore_index = True)

def returnFieldFromFilename(filename): 
    """
    returns field from filename

    Parameters
    ----------
    filename : str
    
    Returns
    -------
    field: float

    """
    if filename[0] == 'n': 
        return np.round(float(file[1:file.find('_')]), 3)
    elif filename[0].isnumeric() or filename[0] == '-': 
        return np.round(float(file[:file.find('_')]), 3)
    else: 
        return 0

    
# wkdir = r'C:\Data\3D Skyrmion\NdMn2Ge2\NdMn2Ge2_2D_DMI_1e-3_relax'
# files = sorted([file for file in os.listdir(wkdir) if file.find('ovf') != -1 or file.find('omf') != -1])

# for i, file in enumerate(files[:1]):
#     print(i)
#     data = mumax(os.path.join(wkdir, file))
#     data.findAndCharacterizeSkyrmions()
#     field = returnFieldFromFilename(file)
#     if i == 0: 
#         skDatabase = data.skyrmionAnalysis
#         skDatabase['field'] = field
#     else: 
#         tempDB = data.skyrmionAnalysis
#         tempDB['field'] = field
#         skDatabase = pd.concat([skDatabase, tempDB], ignore_index = True)
        
# # Single        
# # plt.imshow(test.m[2,...,0])
# # sns.scatterplot(data = test.skyrmionAnalysis, x = 'centroid-1', y = 'centroid-0', hue = 'Q', cmap = 'reds')

# # multi
# sns.scatterplot(data = skDatabase, x = 'field', y = 'Q', hue = 'type')