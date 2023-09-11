import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from time import perf_counter

def timeOperation(func): 
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        total_time = perf_counter()- start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def ims2Array(homedir, file_dir): 
    return np.array([np.array(Image.open(os.path.join(homedir,file_dir,file))) for file in sorted(os.listdir(os.path.join(homedir, file_dir))) if file.find('tif') != -1])

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def dateToSave(time = False): 
    """Produces string of the time in YYYYMMDD format
    if time string includes time in HHMM """
    
    import datetime
    if time: 
        return datetime.datetime.now().strftime('%Y%m%d_%H%M')
    else: 
        return datetime.datetime.now().strftime('%Y%m%d')

def reduceArraySize(array, thresh = 0.1, buffer = 5): 
    p = np.where(abs(array) > np.amax(abs(array))*thresh)
    bounds = np.array([[min(p[0]),max(p[0])], [min(p[1]),max(p[1])], [min(p[2]),max(p[2])]])
    new = np.zeros(shape = (bounds[:,1] - bounds[:,0] + 2*buffer), dtype = array.dtype)
    new[buffer:-buffer, buffer:-buffer, buffer:-buffer] = array[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1], bounds[2,0]:bounds[2,1]]
    return new 

#levi-civita
def E(i,j,k): 
    return int((i-j)*(j-k)*(k-i)/2)

def centreArray(array): 
    if len(array.shape) == 3: 
        x, y, z = center_of_mass(abs(array))
        xc, yc, zc = array.shape
        arr = np.copy(array, order = "C")
        i = 0
        for com, cen in zip([x,y,z], [xc,yc,zc]):
            arr = np.roll(arr, -int(np.round(com-cen/2, 0)), axis = i)
            i += 1
    else: 
        x, y = center_of_mass(abs(array))
        xc, yc = array.shape
        arr = np.copy(array, order = "C")
        i = 0
        for com, cen in zip([x,y], [xc,yc]):
            arr = np.roll(arr, -int(np.round(com-cen/2, 0)), axis = i)
            i += 1
    return arr

def arrayThresh(rec_dict, threshDict, scans): 
    arraySize = np.zeros(shape = (len(scans), 3))
    for i in range(len(scans)):
        arraySize[i, :] = rec_dict['{}'.format(scans[i])].shape
    rec_dict_new = {}
    mask_dict_new = {}
    for scan in scans:
        arr = rec_dict['{}'.format(scan)]
        new = np.zeros(shape = (int(max(arraySize[:,0])+1), int(max(arraySize[:,1])+1), int(max(arraySize[:,2]))+1), dtype = arr.dtype)
        new[1:(1+arr.shape[0]), 1:(1+arr.shape[1]), 1:(arr.shape[2]+1)] = arr
        new = centreArray(new)
        newMask = abs(new) > threshDict['{}'.format(scan)]*abs(np.amax(new))
        rec_dict_new.update({'{}'.format(scan): new})
        mask_dict_new.update({'{}'.format(scan): newMask})        
    return rec_dict_new, mask_dict_new

from scipy.ndimage import zoom
def zoom2(standard, array2zoom):
    widthRatio = standard.shape[-2]/array2zoom.shape[-2]
    heightRatio = standard.shape[-1]/array2zoom.shape[-1]
    if standard.ndim == 4: 
        new = np.zeros_like(standard)
        for i in range(new.shape[0]): 
            for j in range(new.shape[1]): 
                new[i,j] = zoom(array2zoom[i, j], (widthRatio, heightRatio))
    elif standard.ndim == 3: 
        new = np.zeros_like(standard)
        for i in range(new.shape[0]):  
            new[i] = zoom(array2zoom[i], (widthRatio, heightRatio))
    return new

def makeFolder(folderName, savedir): 
    path = os.path.join(savedir, folderName)
    if os.path.exists(path) == False: 
        os.mkdir(path)
        os.chdir(path)
    else: 
        os.chdir(path)
        
def imageViewer(array, sliceNo, direction): 
    """
    

    Parameters
    ----------
    array : array 
    sliceNo : sliceNo to view
    direction : 'x', 'y' or 'z'

    Returns
    -------
    image

    """
    a = np.copy(array, order = "C")
    import matplotlib.pyplot as plt
    if a.dtype == 'complex64': 
        a = abs(a)
    if direction == 'x':
        plt.imshow(np.swapaxes(a[sliceNo], 0,1))
    elif direction == 'y':
        plt.imshow(np.swapaxes(a[:, sliceNo, :], 0,1))
    elif direction == 'z':
        plt.imshow(a[..., sliceNo])
        
def ScaleBarCalculator(x, y, reference, scaleBarSize): 
    d = np.sqrt(x**2 + y**2) #per mm
    ratio = d/reference
    return ratio*scaleBarSize

def FFTFilter(array, sliceToFilter, component, r, passType = 'high'): 
    array = np.copy(array[component, ..., sliceToFilter])
    init = np.fft.fftshift(np.fft.fftn(array))
    xx, yy = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    cen = [int(array.shape[1]/2), int(array.shape[0]/2)]
    rad = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) < r
    "low pass filter"
    if passType == 'low': 
        init[~rad] = 0
    elif passType == 'high': 
        init[rad] = 0
    else: 
        print('passType must be either high or low')
    ifft = np.fft.ifftn(init)
    return ifft

def standardDeviationMap(array, window): 
    stddev = np.zeros_like(array)
    for i in range(window, array.shape[0]-window): 
        for j in range(window, array.shape[1]-window): 
            stddev[i,j] = np.std(array[i-window:i+window, j-window: j+window])
    return stddev

def circle(array, size, centre = None): 
    if centre == None: 
        centre = [int(array.shape[0]/2), int(array.shape[1]/2)]
    yy, xx = np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]))
    yy = yy - centre[0]
    xx = xx - centre[1]
    mask = np.sqrt((xx)**2 + yy**2) < size
    return mask