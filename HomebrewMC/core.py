# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:16:18 2023

@author: massey_j
"""

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

def calculateDEnergy2(spins, fourSpinTranslations): 
    #Will give PBCS
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                spini = spins[:,i,j,k]
                total = 0
                for p in fourSpinTranslations: 
                    if i == 0 or i == spins.shape[1] -1 or j == 0 or j == spins.shape[2] -1 or k == 0 or k == spins.shape[3] -1:
                        ptemp = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in p]
                    else: 
                        ptemp = [np.array([i,j,k]) + np.array(translation) for translation in p]
                    spinj = spins[:,ptemp[0][0], ptemp[0][1], ptemp[0][2]]
                    spink = spins[:,ptemp[1][0], ptemp[1][1], ptemp[1][2]]
                    spinl = spins[:,ptemp[2][0], ptemp[2][1], ptemp[2][2]]
                    total += (1/3)*(np.dot(spini, spinj)*np.dot(spink, spinl) + 
                                    np.dot(spini, spinl)*np.dot(spinj, spink) + 
                                    np.dot(spini, spink)*np.dot(spinj, spinl))
                DEn[i,j,k] = total/len(fourSpinTranslations)
    return DEn

def calculateDEnergy2WithPBCsOff(spins, fourSpinTranslations, zPBCsOff = False): 
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                spini = spins[:,i,j,k]
                total = 0
                missed = 0
                for p in fourSpinTranslations: 
                    if zPBCsOff: 
                        if i == 0 or i == spins.shape[1] -1 or j == 0 or j == spins.shape[2] -1 or k == 0 or k == spins.shape[3] -1:
                            if [any(np.array(translation) + np.array([i,j,k]) >= [spins.shape[1],spins.shape[2],spins.shape[3]]) for translation in p][2]: 
                                missed += 1
                                ptemp = None
                            elif [any(np.array(translation) + np.array([i,j,k]) < [0,0,0]) for translation in p][2]: 
                                missed += 1
                                ptemp = None
                            else: 
                                ptemp = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in p]
                            
                    else: 
                        if i == 0 or i == spins.shape[1] -1 or j == 0 or j == spins.shape[2] -1 or k == 0 or k == spins.shape[3] -1:
                            ptemp = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in p]
                        else: 
                            ptemp = [np.array([i,j,k]) + np.array(translation) for translation in p]
                    if ptemp:
                        spinj = spins[:,ptemp[0][0], ptemp[0][1], ptemp[0][2]]
                        spink = spins[:,ptemp[1][0], ptemp[1][1], ptemp[1][2]]
                        spinl = spins[:,ptemp[2][0], ptemp[2][1], ptemp[2][2]]
                        total += (1/3)*(np.dot(spini, spinj)*np.dot(spink, spinl) + 
                                        np.dot(spini, spinl)*np.dot(spinj, spink) + 
                                        np.dot(spini, spink)*np.dot(spinj, spinl))
                DEn[i,j,k] = total/(len(fourSpinTranslations)-missed)
    return DEn

def initializeTranslations(): 
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

def calculateTotalSpinEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations): 
    return np.sum(J1*calculateJEnergyOverNN(spins, NNTranslations) + J2*calculateJEnergyOverNN(spins, NNNTranslations) + D*calculateDEnergy2(spins, fourSpinTranslations)), np.sum(J1*calculateJEnergyOverNN(spins, NNTranslations)), np.sum(J2*calculateJEnergyOverNN(spins, NNNTranslations)), np.sum(D*calculateDEnergy2(spins, fourSpinTranslations))
"""Turn this into a single loop that gives the same thing, looping through everything 3 times here"""

@timeOperation
def calculateTotalSpinEnergy2(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations): 
    JEnNN = np.zeros(shape = spins.shape[1:])
    JEnNNN = np.zeros(shape = spins.shape[1:])
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                #redo translations
                translatedPositionsNN = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in NNTranslations]
                translatedPositionsNNN = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in NNNTranslations]
                translatedPositionsFourSpin = [[[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(translation)), spins.shape[1:])] for translation in p] for p in fourSpinTranslations]
                #NN 
                totalNN, totalNNN, totalFourSpin = 0,0,0
                for translatedPosition in translatedPositionsNN:
                    totalNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEnNN[i,j,k] = 0.5*totalNN/len(NNTranslations)
                
                #NNN
                for translatedPosition in translatedPositionsNNN:
                    totalNNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEnNNN[i,j,k] = 0.5*totalNNN/len(NNNTranslations)
                
                for translatedPositionGroup in translatedPositionsFourSpin: 
                    totalFourSpin += (1/3)*(np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]])*np.dot(spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]))
                DEn[i,j,k] = totalFourSpin/len(translatedPositionsFourSpin)
    return J1*np.sum(JEnNN) + J2*np.sum(JEnNNN) + D*np.sum(DEn), J1*np.sum(JEnNN), J2*np.sum(JEnNNN), D*np.sum(DEn)


def calculateTotalSpinEnergy3(spins, J1, J2, D, translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin): 
    JEnNN = np.zeros(shape = spins.shape[1:])
    JEnNNN = np.zeros(shape = spins.shape[1:])
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                #NN 
                totalNN, totalNNN, totalFourSpin = 0,0,0
                for translatedPosition in translatedPositionsNN[i,j,k]:
                    totalNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEnNN[i,j,k] = 0.5*totalNN/len(translatedPositionsNN[i,j,k])
                
                #NNN
                for translatedPosition in translatedPositionsNNN[i,j,k]:
                    totalNNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEnNNN[i,j,k] = 0.5*totalNNN/len(translatedPositionsNNN[i,j,k])
                
                for translatedPositionGroup in translatedPositionsFourSpin[i,j,k]: 
                    totalFourSpin += (1/3)*(np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]])*np.dot(spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]))
                DEn[i,j,k] = totalFourSpin/len(translatedPositionsFourSpin[i,j,k])
    return J1*np.sum(JEnNN) + J2*np.sum(JEnNNN) + D*np.sum(DEn), J1*np.sum(JEnNN), J2*np.sum(JEnNNN), D*np.sum(DEn)

def calculateTotalSpinEnergy4(spins, J1, J2, D, totalTranslations, translatedPositionsNN, translatedPositionsNNN, 
                              translatedPositionsFourSpin, coordinationNumberToRemove): 
    #For the coordination number calculations
    JEnNN = np.zeros(shape = spins.shape[1:])
    JEnNNN = np.zeros(shape = spins.shape[1:])
    DEn = np.zeros(shape = spins.shape[1:])
    for i in range(spins.shape[1]): 
        for j in range(spins.shape[2]):
            for k in range(spins.shape[3]):
                #NN 
                NNToRemove, NNNToRemove, DToRemove = 0,0,0
                spinsToRemove = [[PBCChecker(res,size) for res,size in zip((np.array([i,j,k]) + np.array(totalTranslations[translation])), spins.shape[1:])] for translation in np.random.randint(0, len(totalTranslations),coordinationNumberToRemove)]
                totalNN, totalNNN, totalFourSpin = 0,0,0
                NNToRemove, NNNToRemove, DToRemove = 0,0,0
                for translatedPosition in translatedPositionsNN[i,j,k]: 
                    if any([all(s2r == translatedPosition) for s2r in spinsToRemove]): 
                        NNToRemove += 1
                    else:
                        totalNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                        
                JEnNN[i,j,k] = 0.5*totalNN/(len(translatedPositionsNN[i,j,k]) - NNToRemove)
                
                #NNN
                for translatedPosition in translatedPositionsNNN[i,j,k]:
                    if any([all(s2r == translatedPosition) for s2r in spinsToRemove]): 
                        NNNToRemove += 1
                    else:
                        totalNNN += np.dot(spins[:,i,j,k], spins[:,translatedPosition[0],translatedPosition[1],translatedPosition[2]])
                JEnNNN[i,j,k] = 0.5*totalNNN/(len(translatedPositionsNNN[i,j,k]) - NNNToRemove)
                
                for translatedPositionGroup in translatedPositionsFourSpin[i,j,k]: 
                    if any([any([all(s2r == translatedPosition) for s2r in spinsToRemove]) for translatedPosition in translatedPositionGroup]):
                        DToRemove += 1
                    else: 
                        totalFourSpin += (1/3)*(np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]])*np.dot(spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]]) + 
                                    np.dot(spins[:,i,j,k], spins[:,translatedPositionGroup[1][0], translatedPositionGroup[1][1], translatedPositionGroup[1][2]])*np.dot(spins[:,translatedPositionGroup[0][0], translatedPositionGroup[0][1], translatedPositionGroup[0][2]], spins[:,translatedPositionGroup[2][0], translatedPositionGroup[2][1], translatedPositionGroup[2][2]]))
                DEn[i,j,k] = totalFourSpin/(len(translatedPositionsFourSpin[i,j,k]) - DToRemove)
    return J1*np.sum(JEnNN) + J2*np.sum(JEnNNN) + D*np.sum(DEn), J1*np.sum(JEnNN), J2*np.sum(JEnNNN), D*np.sum(DEn)

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
 

def flipRandomSpin(spins): 
    x,y,z = np.unravel_index(np.random.randint(0, spins.shape[1]*spins.shape[2]*spins.shape[3]), spins.shape[1:])
    theta = np.random.rand()*2*np.pi 
    phi= np.random.rand()*2*np.pi
    tempSpins = np.copy(spins, order = 'C')
    tempSpins[:,x,y,z] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return tempSpins


def runMCSimSpin(spins, temperature, iterations, J1, J2, D, 
             NNTranslations, NNNTranslations, fourSpinTranslations,
             preIterations, kB): 
    #Initialize
    energyTrace = []
    totalSpinE = calculateTotalSpinEnergy(spins, J1, J2, D, NNTranslations, NNNTranslations, fourSpinTranslations)
    #Move and test
    count = 1
    print(f'Initializing simulation at {temperature} K')
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
            if count: 
                print(f'Beginning main body of {iterations} iterations for {temperature} K after {preIterations} initialization steps.')
                count = 0
            energyTrace.append(totalSpinE)
            
    return spins, energyTrace

def runMCSimSpin2(spins, temperature, iterations, J1, J2, D, 
             translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin, 
             preIterations, kB): 
    #Initialize
    energyTrace = []
    totalSpinE, totalEJ1, totalEJ2, totalED  = calculateTotalSpinEnergy3(spins, J1, J2, D, translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin)
    #Move and test
    count = 1
    print(f'Initializing simulation at {temperature} K')
    for i in range(iterations+preIterations):
        testSpins = flipRandomSpin(spins)
        testE, testEJ1, testEJ2, testED = calculateTotalSpinEnergy3(spins, J1, J2, D, translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin)
        EDiff = testE - totalSpinE
        if EDiff < 0: 
            spins = testSpins
            totalSpinE, totalEJ1, totalEJ2, totalED = testE, testEJ1, testEJ2, testED
        else: 
            if np.exp(-(EDiff/kB*temperature)) > np.random.rand(): 
                spins = testSpins
                totalSpinE, totalEJ1, totalEJ2, totalED = testE, testEJ1, testEJ2, testED
        if i > preIterations:
            if count: 
                print(f'Beginning main body of {iterations} iterations for {temperature} K after {preIterations} initialization steps.')
                count = 0
            energyTrace.append([totalSpinE, totalEJ1, totalEJ2, totalED])   
    return spins, energyTrace

def runMCSim_StatisticsOfSingleSpinFlip(spins, temperature, iterations, J1, J2, D, 
             translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin, 
             preIterations, kB): 

    energyTrace = []
    totalSpinE, totalEJ1, totalEJ2, totalED = calculateTotalSpinEnergy3(spins, J1, J2, D, translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin)
    ticker = 0
    for i in range(iterations):
        if i == ticker*(iterations/10): 
            print(f'Through {i} iterations.')
            ticker += 1
        testSpins = flipRandomSpin(spins)
        testE, testEJ1, testEJ2, testED = calculateTotalSpinEnergy3(testSpins, J1, J2, D, translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin)
        energyTrace.append([testE - totalSpinE, testEJ1-totalEJ1, testEJ2 - totalEJ2, testED - totalED])
            
    return energyTrace

def runMCSim_StatisticsOfSingleSpinFlipWithCoordinationNumber(spins, J1, J2, D, totalTranslations,
    translatedPositionsNN, translatedPositionsNNN, translatedPositionsFourSpin, coordinationNumberToRemove, iterations): 

    energyTrace = []
    totalSpinE, totalEJ1, totalEJ2, totalED = calculateTotalSpinEnergy4(spins, J1, J2, D, totalTranslations, translatedPositionsNN, translatedPositionsNNN, 
                                  translatedPositionsFourSpin, coordinationNumberToRemove)
    ticker = 0
    for i in range(iterations):
        if i == ticker*(iterations/10): 
             print(f'Through {i} iterations.')
             ticker += 1
        testSpins = flipRandomSpin(spins)
        testE, testEJ1, testEJ2, testED = calculateTotalSpinEnergy4(testSpins, J1, J2, D, totalTranslations, translatedPositionsNN, translatedPositionsNNN, 
                                      translatedPositionsFourSpin, coordinationNumberToRemove)
        energyTrace.append([testE - totalSpinE, testEJ1-totalEJ1, testEJ2 - totalEJ2, testED - totalED])
    return energyTrace

def initializeAFState(xshape, yshape, zshape, axis): 
    spins = np.zeros(shape = (3, xshape, yshape, zshape))
    spins[axis,::2,::2,::2] = 1
    spins[axis,::2,1::2,::2] = -1
    spins[axis,1::2,::2,::2] = -1
    spins[axis,1::2,1::2,::2] = 1
    spins[axis,::2,::2,1::2] = -1
    spins[axis,::2,1::2,1::2] = 1
    spins[axis,1::2,::2,1::2] = 1
    spins[axis,1::2,1::2,1::2] = -1
    return spins

def initializeFMState(xshape, yshape, zshape, axis): 
    spins = np.zeros(shape = (3, xshape, yshape, zshape))
    spins[axis] = 1
    return spins

                                
direction = lambda num: 'x' if num == 0 else 'y' if num == 1 else 'z'

kB = 1.38E-23