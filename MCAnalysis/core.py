# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:40:48 2023

@author: massey_j
"""
import numpy as np
from pyJM.BasicFunctions import find_all
from scipy.optimize import curve_fit
import os

"""Translations and defect identifiers"""
def initializeTranslations(): 
    """
    initializes transition arrays for: 
        nearest neighbour
        next nearest neighbour
        four-spin quartets 

    Returns
    -------
    NNTranslations : list
    NNNTranslations : list
    fourSpinTranslations : list

    """
    NNTranslations = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    NNNTranslations = [[-1, -1, 0],
     [-1, 0, -1],
     [-1, 0, 1],
     [-1, 1, 0],
     [0, -1, -1],
     [0, -1, 1],
     [0, 1, -1],
     [0, 1, 1],
     [1, -1, 0],
     [1, 0, -1],
     [1, 0, 1],
     [1, 1, 0]]
    fourSpinTranslations = [[[1,0,0], [0,1,0], [0,0,1]], 
                                [[-1,0,0],[0,1,0],[0,0,1]], 
                                [[1,0,0 ],[0,-1,0 ],[0,0,1]], 
                                [[1,0,0], [0,1,0], [0,0,-1 ]], 
                                [[-1,0,0 ], [0,-1,0 ],[0,0,1 ]], 
                                [[-1,0,0 ],[0,1,0], [0,0,-1 ]], 
                                [[1,0,0 ],[0,-1,0 ],[0,0,-1 ]],
                                [[-1,0,0 ], [0,-1,0 ], [0,0,-1 ]], 
                                [[1,0,0 ], [1,1,0 ], [1,0,1 ]], 
                                [[1,0,0 ], [1,-1,0 ], [1,0,1 ]],
                                [[1,0,0 ], [1,1,0 ], [1,0,-1 ]],
                                [[1,0,0 ], [1,-1,0 ], [1,0,-1 ]], 
                                [[-1,0,0 ], [-1,1,0 ], [-1,0,1 ]], 
                                [[-1,0,0 ], [-1,-1,0 ], [-1,0,1 ]],
                                [[-1,0,0 ], [-1,1,0 ], [-1,0,-1 ]],
                                [[-1,0,0 ], [-1,-1,0 ], [-1,0,-1 ]],
                                [[0,1,0 ], [1,1,0 ], [0,1,1 ]], 
                                [[0,1,0 ], [-1,1,0 ], [0,1,1 ]], 
                                [[0,1,0 ], [1,1,0 ], [0,1,-1 ]],
                                [[0,1,0 ], [-1,1,0 ], [0,1,-1 ]], 
                                [[0,-1,0 ], [1,-1,0 ], [0,-1,1 ]], 
                                [[0,-1,0 ], [-1,-1,0 ], [0,-1,1 ]],
                                [[0,-1,0 ], [1,-1,0 ], [0,-1,-1 ]], 
                                [[0,-1,0 ], [-1,-1,0 ], [0,-1,-1 ]], 
                                [[0,0,1 ], [1,0,1 ], [0,1,1 ]], 
                                [[0,0,1 ], [-1,0,1 ], [0,1,1 ]], 
                                [[0,0,1 ], [1,0,1 ], [0,-1,1 ]], 
                                [[0,0,1 ], [-1,0,1 ], [0,-1,1 ]], 
                                [[0,0,-1 ], [1,0,-1 ], [0,1,-1 ]], 
                                [[0,0,-1 ], [-1,0,-1 ], [0,1,-1 ]], 
                                [[0,0,-1 ], [1,0,-1 ], [0,-1,-1 ]], 
                                [[0,0,-1 ], [-1,0,-1 ], [0,-1,-1 ]]]
    
    
    return NNTranslations, NNNTranslations, fourSpinTranslations

def PBCChecker2(res, size, PBC): 
    """
    handles whether spins must be considered 
    in a calculation given periodic boundary conditions

    Parameters
    ----------
    res : list
        result of transition to check.
    size : list
        size of obhect.
    PBC : Bool
        True or false PBC on.

    Returns
    -------
    positionOfSpin considering PBC: list

    """
    #upper bound
    if res >= size:
        if PBC: 
            return res-size
        else: 
            return -5
    #lower bound
    elif res < 0:
        if PBC: 
            return res
        else: 
            return -5
    else: 
        return res
    
def defectChecker(r, defects): 
    """
    checks if position r is in defects

    Parameters
    ----------
    r : list
        position.
    defects : list
        list of defect position.

    Returns
    -------
    temp : Bool
        Defect or not.

    """
    temp = False
    for d in defects: 
        if all(r == d): 
            temp = True
    return temp
    
def translationChecker(i,j,k, translationSet, spin, PBCs, defects): 
    """
    checks if i,j,k + translationSet is out of bounds iwth pbc
    

    Parameters
    ----------
    i : int
        coord x.
    j : int
        coord y.
    k : int 
        coord z.
    translationSet : list
        list of translations.
    spin : np.array
        spin vector of spins[i,j,k].
    PBCs : Bool
        PBCS on or off.
    defects : list
        list of defects.

    Returns
    -------
    coordNo : int
        coord number in transition set.
    toReturn : list
        check if any are out of bounds considering PBC.

    """
    toReturn = [any([PBCChecker2(res,size,PBC) == -5 for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spin.shape[1:],PBCs)]) for translation in translationSet]
    result = [[PBCChecker2(res,size,PBC) for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spin.shape[1:],PBCs)] for translation in translationSet]
    defectReturn = [defectChecker(r,defects) for r in result]
    toReturn = np.logical_or(toReturn, defectReturn)
    coordNo = len(translationSet) - sum(toReturn)
    toReturn = [res for res,r in zip(result, toReturn) if r == False]
    while len(toReturn) < len(translationSet): 
        toReturn.append([0,0,0])
    return coordNo, toReturn

"""Sorting and loading functions"""
def returnDFromFileName(filename): 
    """

    Parameters
    ----------
    filename : str

    Returns
    -------
    D: str

    """
    dashes = list(find_all(filename, '_'))
    return filename[dashes[-1] + 1:-1]

def sortByD(files): 
    """
    

    Parameters
    ----------
    files : str

    Returns
    -------
    sortedList: list

    """
    D = [returnDFromFileName(file) for file in files]
    return list(np.array(files)[np.argsort(D)])

def returnIterationsFromFileName(filename): 
    """

    Parameters
    ----------
    filename : str

    Returns
    -------
    iterations: str

    """
    dashes = list(find_all(filename, '_'))
    steps = list(find_all(filename, 'Steps'))
    return filename[dashes[-1] + 1:steps[-1]]

def sortByIterations(files): 
    """
    

    Parameters
    ----------
    files : str

    Returns
    -------
    sortedList: list

    """
    itNum = [int(returnIterationsFromFileName(file)) for file in files]
    return list(np.array(files)[np.argsort(itNum)])

def returnIterationsFromFileName2(filename): 
    """
    for revised filename protocol
    Parameters
    ----------
    filename : str

    Returns
    -------
    iterations: str

    """
    steps = list(find_all(filename, 'Steps'))
    return filename[steps[0]+6:steps[1]]

def sortByIterations2(files): 
    """
    

    Parameters
    ----------
    files : str

    Returns
    -------
    sortedList: list

    """
    itNum = [int(returnIterationsFromFileName2(file)) for file in files]
    return list(np.array(files)[np.argsort(itNum)])

def returnVacancyFromFilename(filename): 
    """

    Parameters
    ----------
    filename : str

    Returns
    -------
    vacancy: str

    """
    return filename[list(find_all(filename, 'Start'))[0] + 6:list(find_all(filename, 'Vacancy'))[0]]

def sortByVacancies(files): 
    """
    

    Parameters
    ----------
    files : str

    Returns
    -------
    sortedList: list

    """
    vacNum = [returnVacancyFromFilename(file) for file in files]
    return list(np.array(files)[np.argsort(vacNum)])

def returnLocalDFromFilename(filename): 
    """

    Parameters
    ----------
    filename : str

    Returns
    -------
    local D: str

    """
    return filename[filename.find('LocalD') + 7:]

def sortByLocalD(files): 
    """
    

    Parameters
    ----------
    files : str

    Returns
    -------
    sortedList: list

    """
    vacNum = [returnLocalDFromFilename(file) for file in files]
    return list(np.array(files)[np.argsort(vacNum)])

def sortDir(directory, heatPath, coolPath, zPBC, option): 
    """
    sorted files in heatPath and coolPath by option into 
    zPBC/non ZPBC

    Parameters
    ----------
    directory : str
        folder
    heatPath : str
        descriptor of heat path.
    coolPath : str
        descriptor of cool path..
    zPBC : str
        zPBC flag.
    option : str
        type of experiment. should be one of 
        'Vacancies', 'LocalD', 'Iterations', 'Iterations2','UniversalD'

    Returns
    -------
    heat : list
        ordered list.
    heat_zPBC : list
        ordered list.
    cool : list
        ordered list.
    cool_zPBC : list
        ordered list.

    """
    if option == 'Vacancies': 
        sortFunction = sortByVacancies
    elif option == 'LocalD': 
        sortFunction = sortByLocalD
    elif option == 'Iterations': 
        sortFunction = sortByIterations
    elif option == 'Iterations2': 
        sortFunction = sortByIterations2
    elif option == 'UniversalD': 
        sortFunction = sortByD
    else: 
        print('Sorting Function unspecified, please use a relevant option to continue.')
    heatAll = [file for file in os.listdir(directory) if file.find(heatPath) != -1]
    heat = sortFunction([file for file in heatAll if file.find(zPBC) == -1])
    heat_zPBC = sortFunction([file for file in heatAll if file.find(zPBC) != -1])
    coolAll = [file for file in os.listdir(directory) if file.find(coolPath) != -1]
    cool = sortFunction([file for file in coolAll if file.find(zPBC) == -1])
    cool_zPBC = sortFunction([file for file in coolAll if file.find(zPBC) != -1])
    return heat, heat_zPBC, cool, cool_zPBC


def calcCoordAverage(row): 
    """
    calculates averages across a row

    Parameters
    ----------
    row : pd.series
    
    Returns
    -------
    coord, average, std err: list

    """
    vals = np.array(row[1][::2])
    errs = np.array(row[1][1::2])
    m = (errs/vals > 1) + (vals > 400) + (vals < 300) + (np.isnan(vals)) + (np.isinf(errs))
    return [row[0], np.nanmean(vals[~m]), np.nanstd(vals[~m])/np.sqrt(vals[~m].shape[0])]
    

"""Fitting functions for defect and coordination number analysis"""
# Define the Gaussian function
def Gauss(x, A, x0, sigma):
    """
    gaussian function

    Parameters
    ----------
    x : float
    A : float
    x0 : float
    sigma : float

    Returns
    -------
    y : float

    """
    y = A*np.exp(-(x-x0)**2/(2*sigma**2))
    return y

def fitGauss2(x, y):
    """
    fits gauss to x,y

    Parameters
    ----------
    x : np.array
        x coords
    y : np.array
        y coords.

    Returns
    -------
    [x0 value, error]: list

    """
    p0 = [y[np.argmax(abs(y))], x[np.argmax(abs(y))], 10]
    popt, pcov = curve_fit(Gauss, x, y,p0)
    perr = np.sqrt(np.diag(pcov))
    return [popt[1], perr[1]]

def fitPixelTt(array, i,j,k):
    """
    fit transition temp of given pixel i,j,k

    Parameters
    ----------
    array : np.array
    i : int
    j : int
    k : int

    Returns
    -------
    fitGauss2 result

    """
    if any(np.isnan(array[:,1,i,j,k])):
        return [np.nan, np.nan]
    else: 
        try: 
            x = np.arange(300, 401, 5)
            y = np.gradient(array[:,1,i,j,k])
            return fitGauss2(x,y)
        except:
            return [np.nan, np.nan]
        
        