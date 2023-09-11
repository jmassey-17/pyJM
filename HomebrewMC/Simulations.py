# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:15:21 2023

@author: massey_j
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:56:52 2023

@author: massey_j
"""

import numpy as np
import matplotlib.pyplot as plt
from pyJM.BasicFunctions import timeOperation, dateToSave, find_all
from pyJM.HomebrewMC.core import *
from itertools import combinations, permutations
from time import perf_counter
import os

PBCChecker = lambda res, size: res if res < size else res-size

###For the coordination number for the MC
def PBCChecker2(res, size, PBC): 
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
    
def translationChecker(i,j,k, translationSet, spins, PBCs, defects): 
    toReturn = [any([PBCChecker2(res,size,PBC) == -5 for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:],PBCs)]) for translation in translationSet]
    result = [[PBCChecker2(res,size,PBC) for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:],PBCs)] for translation in translationSet]
    defectReturn = [r in defects for r in result]
    toReturn = np.logical_or(toReturn, defectReturn)
    coordNo = len(translationSet) - sum(toReturn)
    toReturn = [res for res,r in zip(result, toReturn) if r == False]
    while len(toReturn) < len(translationSet): 
        toReturn.append([0,0,0])
    return coordNo, toReturn

defects = [[0,0,0],[4,0,0],[1,0,9]]
NNTranslations, NNNTranslations, fourSpinTranslations = initializeTranslations()
spins = np.zeros(shape = (3,5,5,10))
coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spins, [True,True,False],defects) for i in range(spins.shape[1]) for j in range(spins.shape[2]) for k in range(spins.shape[3])], dtype = object)
coordinationNumberNN = coordsNN.reshape(spins.shape[1], spins.shape[2], spins.shape[3],2)[...,0].astype(int)
activeNNTranslations = coordsNN.reshape(spins.shape[1], spins.shape[2], spins.shape[3],2)[...,1]

coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spins, [True,True,False],defects) for i in range(spins.shape[1]) for j in range(spins.shape[2]) for k in range(spins.shape[3])], dtype = object)
coordinationNumberNNN = coordsNNN.reshape(spins.shape[1], spins.shape[2], spins.shape[3],2)[...,0].astype(int)
activeNNTranslations = coordsNNN.reshape(spins.shape[1], spins.shape[2], spins.shape[3],2)[...,1]

coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spins, [True,True,False],defects) for translationSet in fourSpinTranslations] for i in range(spins.shape[1]) for j in range(spins.shape[2]) for k in range(spins.shape[3])], dtype = object)
coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spins.shape[1], spins.shape[2], spins.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3

activeFourSpinTranslations = coordsFourSpin.reshape(spins.shape[1], spins.shape[2], spins.shape[3],len(fourSpinTranslations),2)[...,1]
###
translations = [
    [-1,0,0], 
    [1,0,0],
    [0,-1,0],
    [0,1,0], 
    [0,0,-1], 
    [0,0,1]]

class simulationRunner:
    def __init__(self, xshape, yshape, zshape, J1, J2, D, saveDir):
        self.xshape, self.yshape, self.zshape = xshape, yshape, zshape
        self.initializeSpins()
        self.NNTranslations, self.NNNTranslations, self.fourSpinTranslations = initializeTranslations()
        self.translatedPositionsNN = np.array([[[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), self.spins.shape[1:])] for translation in self.NNTranslations] for i in range(self.spins.shape[1]) for j in range(self.spins.shape[2]) for k in range(self.spins.shape[3])]).reshape((self.spins.shape[1], self.spins.shape[2], self.spins.shape[3], len(self.NNTranslations), 3))
        self.translatedPositionsNNN = np.array([[[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), self.spins.shape[1:])] for translation in self.NNNTranslations] for i in range(self.spins.shape[1]) for j in range(self.spins.shape[2]) for k in range(self.spins.shape[3])]).reshape((self.spins.shape[1], self.spins.shape[2], self.spins.shape[3], len(self.NNNTranslations), 3))
        self.translatedPositionsFourSpin = np.array([[[[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), self.spins.shape[1:])] for translation in p] for p in self.fourSpinTranslations] for i in range(self.spins.shape[1]) for j in range(self.spins.shape[2]) for k in range(self.spins.shape[3])]).reshape((self.spins.shape[1], self.spins.shape[2], self.spins.shape[3], len(self.fourSpinTranslations), 3, 3))
        self.J1, self.J2, self.D = J1,J2,D
        self.saveDir = saveDir
        if os.path.exists(self.saveDir) ==False: 
            os.mkdir(self.saveDir)
        
    def initializeSpins(self, FMSpins = None, AFSpins = None, axis = None):
        if FMSpins == None and AFSpins == None:
            self.spins = (np.random.random(size = (3, self.xshape, self.yshape, self.zshape))-0.5)/0.5
            self.spins = self.spins/np.sqrt(np.sum(self.spins**2, axis = 0))
        elif FMSpins: 
            self.spins = initializeFMState(self.xshape, self.yshape, self.zshape, axis)
        elif AFSpins: 
            self.spins = initializeAFState(self.xshape, self.yshape, self.zshape, axis)
    
    @timeOperation  
    def runMCSim(self, preIterations, iterations, temperatureRange, simDir, 
                 reinitializeSpins = True, FMSpins = None, AFSpins = None, axis = None): 
        if os.path.exists(os.path.join(self.saveDir, simDir)) == False: 
            os.mkdir(os.path.join(self.saveDir, simDir))
            
        for temperature in temperatureRange:
            if reinitializeSpins: 
                self.initializeSpins(FMSpins, AFSpins, axis)
            initialSpins = self.spins
            simulationParameters = {
                'spins': self.spins,
                'temperature': temperature, 
                'iterations': iterations,
                'J1': self.J1, 
                'J2': self.J2,
                'D': self.D, 
                'translatedPositionsNN': self.translatedPositionsNN,
                'translatedPositionsNNN': self.translatedPositionsNNN,
                'translatedPositionsFourSpin': self.translatedPositionsFourSpin,
                'preIterations': preIterations, 
                'kB': kB, 
                }
            self.spins, energyTrace = runMCSimSpin2(**simulationParameters) 
            
            folderName = f'{temperature}K'
            if os.path.exists(os.path.join(self.saveDir, simDir, folderName)) == False: 
                os.mkdir(os.path.join(self.saveDir, simDir, folderName))
                
            toSave = {
                'result' : self.spins, 
                'initial': initialSpins, 
                'energyTrace': energyTrace,
                }
            for key in list(toSave.keys()): 
                filePath = f'{key}_{dateToSave()}.npy'
                np.save(f'{os.path.join(self.saveDir, simDir, folderName, filePath)}', toSave[key]) 
    
    def runEChangePerSpinFlipSim(self, iterations, preIterations, temperature, simDir, reinitializeSpins = True, FMSpins = None, AFSpins = None, axis = None): 
        if os.path.exists(os.path.join(self.saveDir, simDir)) == False: 
            os.mkdir(os.path.join(self.saveDir, simDir))
        if reinitializeSpins: 
            self.initializeSpins(FMSpins, AFSpins, axis)
        initialSpins = self.spins
        simulationParameters = {
            'spins': self.spins,
            'temperature': temperature, 
            'iterations': iterations,
            'J1': self.J1, 
            'J2': self.J2,
            'D': self.D, 
            'translatedPositionsNN': self.translatedPositionsNN,
            'translatedPositionsNNN': self.translatedPositionsNNN,
            'translatedPositionsFourSpin': self.translatedPositionsFourSpin,
            'preIterations': preIterations, 
            'kB': kB, 
            }
            
        energyTrace = runMCSim_StatisticsOfSingleSpinFlip(**simulationParameters)
            
        folderName = f'{temperature}K'
        if os.path.exists(os.path.join(self.saveDir, simDir, folderName)) == False: 
            os.mkdir(os.path.join(self.saveDir, simDir, folderName))
            
        toSave = {
            'initial': initialSpins, 
            'energyTrace': energyTrace,
            }
        
        for key in list(toSave.keys()): 
            filePath = f'{key}_{dateToSave()}.npy'
            np.save(f'{os.path.join(self.saveDir, simDir, folderName, filePath)}', toSave[key]) 
            
    def runEChangePerSpinFlipVaryingCoordinationNumber(self, iterations, simDir, coordinationNumberToRemove, reinitializeSpins = True, FMSpins = None, AFSpins = None, axis = None): 
        if os.path.exists(os.path.join(self.saveDir, simDir)) == False: 
            os.mkdir(os.path.join(self.saveDir, simDir))
        if reinitializeSpins: 
            self.initializeSpins(FMSpins, AFSpins, axis)
            
        initialSpins = self.spins
        simulationParameters = {
            'spins': self.spins,
            'J1': self.J1, 
            'J2': self.J2,
            'D': self.D, 
            'totalTranslations': self.NNTranslations + self.NNNTranslations,
            'translatedPositionsNN': self.translatedPositionsNN,
            'translatedPositionsNNN': self.translatedPositionsNNN,
            'translatedPositionsFourSpin': self.translatedPositionsFourSpin,
            'coordinationNumberToRemove': coordinationNumberToRemove,
            'iterations': iterations
            }
        energyTrace = runMCSim_StatisticsOfSingleSpinFlipWithCoordinationNumber(**simulationParameters)
            
        folderName = f'{temperature}K'
        if os.path.exists(os.path.join(self.saveDir, simDir, folderName)) == False: 
            os.mkdir(os.path.join(self.saveDir, simDir, folderName))
            
        toSave = {
            'initial': initialSpins, 
            'energyTrace': energyTrace,
            }
        
        for key in list(toSave.keys()): 
            filePath = f'{key}_{dateToSave()}.npy'
            np.save(f'{os.path.join(self.saveDir, simDir, folderName, filePath)}', toSave[key]) 