# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:56:52 2023

@author: massey_j
"""

import numpy as np
import matplotlib.pyplot as plt
from pyJM.BasicFunctions import timeOperation, dateToSave
from itertools import combinations, permutations
from time import perf_counter
import os

PBCChecker = lambda res, size: res if res < size else res-size

translations = [
    [-1,0,0], 
    [1,0,0],
    [0,-1,0],
    [0,1,0], 
    [0,0,-1], 
    [0,0,1]]

def initializeDCalculationTranslations(spins): 
    X, Y, Z = np.meshgrid(np.arange(spins.shape[1]), np.arange(spins.shape[2]), np.arange(spins.shape[3]))
    #nn
    i,j,k = int(spins.shape[1]/2), int(spins.shape[2]/2), int(spins.shape[3]/2)
    nn = (X-i)**2 + (Y-j)**2 + (Z-k)**2 == 1
    x,y,z = np.where(nn == True)
    points = [[u,l,p] for u,l,p in zip(x,y,z)]
    
    nnn = (X-i)**2 + (Y-j)**2 + (Z-k)**2 == 2
    xn,yn,zn = np.where(nnn == True)
    pointsnn = [[u,l,p] for u,l,p in zip(xn,yn,zn)]
    final = points + pointsnn
    
    toTest = list(combinations(final, 3))
    toKeep = []
    for row in enumerate(toTest): 
        count_nn, count_nnn = 0,0
        for p in row[1]: 
            if p in points: 
                count_nn += 1
            elif p in pointsnn: 
                count_nnn += 1
            if count_nn == 1 and count_nnn == 2:
                toKeep.append(row[0])
            elif count_nn == 3:
                toKeep.append(row[0])
                
    toTest = [toTest[ind] for ind in toKeep]
    
    toRemove = []
    for row in enumerate(toTest): 
        for p in row[1]: 
            for p1 in row[1]: 
                if any(abs(np.array(p) - np.array(p1)) >= 2):  
                    toRemove.append(row[0])
                elif sum(abs(np.array(p) - np.array(p1))) >= 3: 
                    toRemove.append(row[0])
    finalPoints = np.delete(toTest, np.unique(toRemove), axis = 0)
    
    return [np.array(p)-np.array([i,j,k]) for p in finalPoints]

def initializeJnnCalculationTranslations(spins, NN): 
    X, Y, Z = np.meshgrid(np.arange(spins.shape[1]), np.arange(spins.shape[2]), np.arange(spins.shape[3]))
    #nn
    i,j,k = int(spins.shape[1]/2), int(spins.shape[2]/2), int(spins.shape[3]/2)
    sphere = (X-i)**2 + (Y-j)**2 + (Z-k)**2 == NN
    x,y,z = np.where(sphere == True)
    return [[u-i,l-j,p-k] for u,l,p in zip(x,y,z)]


def calculateJEnergyOverNN(spins, neighbourTranslations): 
    JEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                total = 0
                for translation in neighbourTranslations: 
                    translatedPosition = [PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])]
                    total += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEn[i,j,k] = 0.5*total/len(neighbourTranslations)
    return JEn
            

def calculateDEnergy(spins, fourSpinTranslations): 
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                spini = spins[:,i,j,k]
                total = 0
                for p in fourSpinTranslations: 
                    ptemp = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in p]
                    spinj = spins[:,ptemp[0][0], ptemp[0][1], ptemp[0][2]]
                    spink = spins[:,ptemp[1][0], ptemp[1][1], ptemp[1][2]]
                    spinl = spins[:,ptemp[2][0], ptemp[2][1], ptemp[2][2]]
                    total += (1/3)*(np.dot(spini, spinj)*np.dot(spink, spinl) + 
                                    np.dot(spini, spinl)*np.dot(spinj, spink) + 
                                    np.dot(spini, spink)*np.dot(spinj, spinl))
                DEn[i,j,k] = total/len(fourSpinTranslations)
    return DEn

def initializeTranslations(spins): 
    return initializeJnnCalculationTranslations(spins, 1), initializeJnnCalculationTranslations(spins, 2), initializeDCalculationTranslations(spins)


def calculateTotalSpinEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations): 
    return np.sum(J1*calculateJEnergyOverNN(spins, NNTranslations) + J2*calculateJEnergyOverNN(spins, NNNTranslations) + D*calculateDEnergy(spins, fourSpinTranslations))


def calculateSpinOP(spins, translations): 
    OP = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]):
        for j in range(spins.shape[2]): 
            for k in range(spins.shape[3]):
                for translation in translations: 
                    translatedPosition = [PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])]
                    OP[i,j,k] += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0], translatedPosition[1], translatedPosition[2]])
    OP = OP/len(translations)
    return OP
 
def calculateSpinOPSingle(spins,i,j,k, translations): 
    OP = 0
    for translation in translations:
        translatedPosition = [PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])]
        OP += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0], translatedPosition[1], translatedPosition[2]])
    OP = OP/len(translations)
    return OP
 

def calculateStrainTensor(latticePosition, latticePositionDefault): 
    u = latticePosition-latticePositionDefault
    e = {}
    for i in (0,1,2): 
        for j in (0,1,2): 
            if f'{direction(j)}{direction(i)}' not in list(e.keys()): #stops double counting the tensor elements
                e.update({f'{direction(i)}{direction(j)}': 0.5*(np.gradient(u[i], axis = j) + np.gradient(u[j], axis = i))})
    return e


def calculateEnergyFromStrainTensor(e, diagonalCoeff, nondiagonalCoeff): 
    diagonal = [key for key in list(e.keys()) if key[0] == key[1]]
    nondiagonal = [key for key in list(e.keys()) if key not in diagonal]
    diagonalEnergy = np.zeros_like(e[diagonal[0]])
    nondiagonalEnergy = np.zeros_like(e[diagonal[0]])
    for key in diagonal: 
        diagonalEnergy += e[key]**2
    for key in nondiagonal: 
        nondiagonalEnergy += e[key]**2
    return np.sum(0.5*diagonalCoeff*diagonalEnergy + nondiagonalCoeff*nondiagonalEnergy)

def calculateTotalEnergy(spin, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations, 
                         latticePosition, latticePositionDefault): 
    return calculateTotalSpinEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations) + calculateEnergyFromStrainTensor(calculateStrainTensor(latticePosition, latticePositionDefault), diagonalCoeff, nondiagonalCoeff)


def flipRandomSpin(spins): 
    x,y,z = np.unravel_index(np.random.randint(0, spins.shape[1]*spins.shape[2]*spins.shape[3]), spins.shape[1:])
    theta = np.random.rand()*2*np.pi 
    phi= np.random.rand()*2*np.pi
    tempSpins = np.copy(spins, order = 'C')
    tempSpins[:,x,y,z] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return tempSpins

def flipRandomSpinWithStrain(spins, latticePosition, latticePositionDefault, translations, AFLatticeParameter, FMLatticeParameter): 
    x,y,z = np.unravel_index(np.random.randint(0, spins.shape[1]*spins.shape[2]*spins.shape[3]), spins.shape[1:])
    theta = np.random.rand()*2*np.pi 
    phi= np.random.rand()*2*np.pi
    tempSpins = np.copy(spins, order = 'C')
    tempSpins[:,x,y,z] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    tempLatticePosition = np.copy(latticePosition, order = 'C')
    tempLatticePositionDefault = np.copy(latticePositionDefault, order = 'C')
    if calculateSpinOPSingle(tempSpins, x, y, z, translations) > 0: 
        tempLatticePosition[:,x,y,z] = np.random.normal(FMLatticeParameter, 0.001, spins.shape[0]*1)
        tempLatticePositionDefault[:,x,y,z] = FMLatticeParameter
    else:  
        tempLatticePosition[:,x,y,z] = np.random.normal(AFLatticeParameter, 0.001, spins.shape[0]*1)
        tempLatticePositionDefault[:,x,y,z] = AFLatticeParameter
    return tempSpins, tempLatticePosition, tempLatticePositionDefault

@timeOperation
def runMCSimSpin(spins, temperature, iterations, J1, J2, D, 
             NNTranslations, NNNTranslations, fourSpinTranslations,
             preIterations = 1000, kB = 1.38E-23): 
    #Initialize
    energyTrace = []
    totalSpinE = calculateTotalSpinEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations)
    #Move and test
    for i in range(iterations+preIterations):
        testSpins = flipRandomSpin(spins)
        testE = calculateTotalSpinEnergy(testSpins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations)
        EDiff = testE - totalSpinE
        if EDiff < 0: 
            spins = testSpins
            totalSpinE = testE
        else: 
            if np.exp(-(EDiff/kB*temperature)) > np.random.rand(): 
                spins = testSpins
                totalSpinE = testE
        if i > preIterations: 
            print(f'iteration {i-preIterations} of {iterations}')
            energyTrace.append(totalSpinE)
            
    return spins, energyTrace

                                
direction = lambda num: 'x' if num == 0 else 'y' if num == 1 else 'z'
phaseLatticeParameter = lambda op: 2.998 if op < 0 else 3

"""
Need a grid
Need a spin and a strain
Need a default lattice constant
"""

# latticePosition = np.random.random(size = (3, xshape, yshape, zshape))/1000 + 2.998
# latticePositionDefault = np.zeros(shape = spins.shape)
# latticePositionDefault[:] = 2.998


# """Minimzation of elastic energy"""
# FM = calculateSpinOP(spins, translations) > 0
# latticePositionDefault[:,FM] = 3
# e = calculateStrainTensor(latticePosition, latticePositionDefault)
# en = calculateEnergyFromStrainTensor(e, 1, 1)

J1 = -0.4E-21
J2 = -2.75E-21
D = 0.23E-21
FMLatticeParameter = 3
AFLatticeParameter = 2.998
diagonalCoeff = 1E-14
nondiagonalCoeff = 0 

xshape = 5
yshape = 5
zshape = 5

spins = (np.random.random(size = (3, xshape, yshape, zshape))-0.5)/0.5
spins = spins/np.sqrt(np.sum(spins**2, axis = 0))

NNTranslations, NNNTranslations, fourSpinTranslations = initializeTranslations(spins)
initialSpins = spins

FM = calculateSpinOP(spins, translations) > 0
latticePosition = np.zeros(shape = spins.shape)
latticePosition[:,FM] = np.random.normal(FMLatticeParameter, 0.001, spins.shape[0]*np.sum(FM)).reshape((spins.shape[0], np.sum(FM)))
latticePosition[:,~FM] = np.random.normal(AFLatticeParameter, 0.001, spins.shape[0]*np.sum(~FM)).reshape((spins.shape[0], np.sum(~FM)))
latticePositionDefault = np.zeros(shape = spins.shape)
latticePositionDefault[:, FM] = FMLatticeParameter
latticePositionDefault[:, ~FM] = AFLatticeParameter

@timeOperation
def runMCSimSpinAndStrain(spins, latticePosition, latticePositionDefault, temperature, iterations, 
                   diagonalCoeff, nondiagonalCoeff, FMLatticeParameter, AFLatticeParameter,
                   preIterations = 1000, kB = 1.38E-23): 
    #Initialize
    energyTrace = []
    totalEnergy = calculateTotalEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations, latticePosition, latticePositionDefault)
    #Move and test
    for i in range(iterations+preIterations):
        tempSpins, tempLatticePosition, tempLatticePositionDefault = flipRandomSpinWithStrain(spins, latticePosition, latticePositionDefault, translations, AFLatticeParameter, FMLatticeParameter)
        testE = calculateTotalEnergy(tempSpins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations, tempLatticePosition, tempLatticePositionDefault)
        EDiff = testE - totalEnergy
        if EDiff < 0: 
            spins, latticePosition, latticePositionDefault = tempSpins, tempLatticePosition, tempLatticePositionDefault
            totalEnergy = testE
        else: 
            if np.exp(-(EDiff/kB*temperature)) > np.random.rand(): 
                spins, latticePosition, latticePositionDefault = tempSpins, tempLatticePosition, tempLatticePositionDefault
                totalEnergy = testE
        if i > preIterations: 
            print(f'iteration {i-preIterations} of {iterations}. E: {totalEnergy}')
            energyTrace.append(totalEnergy)
    return spins, latticePosition, latticePositionDefault, energyTrace
   
temperature = 300
iterations = 100  

spins, latticePosition, latticePositionDefault, energyTrace = runMCSimSpinAndStrain(spins, latticePosition, latticePositionDefault, temperature, iterations, diagonalCoeff, nondiagonalCoeff,
                            FMLatticeParameter, AFLatticeParameter, preIterations = 0) 

# SAVEDIR = r"C:\Data\FeRh\MC code\HomemadeMCTests"

# filePath = f'Result_{temperature}K_{xshape}_{yshape}_{zshape}_{iterations}_{dateToSave()}.npy'
# np.save(f'{os.path.join(SAVEDIR, filePath)}', result)
# np.save(f'Initial_{temperature}K_{iterations}_{dateToSave()}.npy', initialSpins)
# np.save(f'EnergyTrace_{temperature}K_{iterations}_{dateToSave()}.npy', energyTrace)

"""Calculate local spin order parameter and change """