import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from time import perf_counter

def timeOperation(func): 
    """
    Decorator to wrap func and return time taken to run it 

    Parameters
    ----------
    func : func
        function to time.

    Returns
    -------
    timeit_wrapper: decorator
        time decorator

    """
    def timeit_wrapper(*args, **kwargs):
        """
        wrapper to time function with *args, **kwargs

        Parameters
        ----------
        *args : 
            arguments for func.
        **kwargs : 
            kwargs for func.

        Returns
        -------
        result : 
            output of func.

        """
        start_time = perf_counter()
        result = func(*args, **kwargs)
        total_time = perf_counter()- start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def ims2Array(homedir, file_dir): 
    """
    takes set of .tif images in homedir/file_dir and collects them into an array

    Parameters
    ----------
    homedir : str
        top dir where the file_dir is stored.
    file_dir : str
        directory where files are stored.

    Returns
    -------
    imageArray: array
        images collected into array.

    """
    return np.array([np.array(Image.open(os.path.join(homedir,file_dir,file))) for file in sorted(os.listdir(os.path.join(homedir, file_dir))) if file.find('tif') != -1])

def find_all(a_str, sub):
    """
    generator object that finds all instances of sub in a_str

    Parameters
    ----------
    a_str : str
        string to test.
    sub : str
        substring to look for.

    Yields
    ------
    start : int
        pos of sub in a_str.

    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def dateToSave(time = False): 
    """
    produces the date in YYYYMMDD(_HHMM) format

    Parameters
    ----------
    time : Bool, optional
        Bool to indicate whether time is also returned. The default is False.

    Returns
    -------
    date(time): str
        date in YYYYMMDD(_HHMM) format.

    """
    """Produces string of the time in YYYYMMDD format
    if time string includes time in HHMM """
    
    import datetime
    if time: 
        return datetime.datetime.now().strftime('%Y%m%d_%H%M')
    else: 
        return datetime.datetime.now().strftime('%Y%m%d')

def reduceArraySize(array, thresh = 0.1, buffer = 5): 
    """
    takes 2d array and reduces it to a size that is +buffer 
    bigger than the last point where data is seen in both directiuons

    Parameters
    ----------
    array : np.array
        array to reduce.
    thresh : float, optional
        magnitude threshold above which information is taken. The default is 0.1.
    buffer : int, optional
        size of buffer zone outside of the area with signal. The default is 5.

    Returns
    -------
    new : np.array
        reduced array.

    """
    p = np.where(abs(array) > np.amax(abs(array))*thresh)
    bounds = np.array([[min(p[0]),max(p[0])], [min(p[1]),max(p[1])], [min(p[2]),max(p[2])]])
    new = np.zeros(shape = (bounds[:,1] - bounds[:,0] + 2*buffer), dtype = array.dtype)
    new[buffer:-buffer, buffer:-buffer, buffer:-buffer] = array[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1], bounds[2,0]:bounds[2,1]]
    return new 

#levi-civita
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

def centreArray(array): 
    """
    Will centre array around its centre of mass

    Parameters
    ----------
    array : np.array

    Returns
    -------
    arr : np.array
        centered array.

    """
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

from scipy.ndimage import zoom
def zoom2(standard, array2zoom):
    """
    zooms array2zoom to be the same size as standard along the -1 axis

    Parameters
    ----------
    standard : np.array
        reference image.
    array2zoom : np.array
        image2zoom.

    Returns
    -------
    new : np.array
        zoomed array2zoom.

    """
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
    """
    makes folder at savedir/folderName

    Parameters
    ----------
    folderName : str
        name of folder to make.
    savedir : str
        directory where folder should be made .

    Returns
    -------
    None.

    """
    path = os.path.join(savedir, folderName)
    if os.path.exists(path) == False: 
        os.mkdir(path)
        os.chdir(path)
    else: 
        os.chdir(path)
        

def FFTFilter(array, sliceToFilter, component, r, passType = 'high'): 
    """
    returns a FFT filtered version of 4d array

    Parameters
    ----------
    array : np.array
        4d numpy array
    sliceToFilter : int
        slice of array to filter
    component : int
        component of first axis in array.
    r : float
        size of the circle mask used to filter.
    passType : str, optional
        type of filter. The default is 'high'.

    Returns
    -------
    ifft : TYPE
        DESCRIPTION.
        
    Raises: 
        ValueError 'passType must be either high or low'

    """
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
        raise ValueError('passType must be either high or low')
    ifft = np.fft.ifftn(init)
    return ifft

def standardDeviationMap(array, window): 
    """
    creates standard deviation map of array looking through window centered around each point 

    Parameters
    ----------
    array : np.array
    window : int
        width of window around point taken to calculate std.

    Returns
    -------
    stddev : np.array
        std of array.

    """
    stddev = np.zeros_like(array)
    for i in range(window, array.shape[0]-window): 
        for j in range(window, array.shape[1]-window): 
            stddev[i,j] = np.std(array[i-window:i+window, j-window: j+window])
    return stddev
