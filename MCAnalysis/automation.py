# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:25:56 2023

@author: massey_j
"""

import numpy as np 
import os
import glob
import shutil
from scipy.optimize import curve_fit
from pyJM.BasicFunctions import find_all
from pyJM.MCAnalysis.core import *

  
NNTranslations, NNNTranslations, fourSpinTranslations = initializeTranslations()
       
def loadAndProcess(file):
    """
    Takes a MC output file and splits it into position (x,y,z), spin (sx, sy, sz), heat capacity (hc)
    and order parameters (AF, FM)
    """
    data = np.genfromtxt(file, skip_header=2)
    x = data[:,0].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    y = data[:,1].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    z = data[:,2].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    sx = data[:,3].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    sy = data[:,4].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    sz = data[:,5].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    hc = data[:,6].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    af = data[:,7].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    fm = data[:,8].reshape((int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1))
    return np.array([x, y, z]), np.array([sx, sy, sz]), hc, np.array([af,fm])
            
def transitionTemperatureCoordinationNumber(wkdir):
    """
    Takes wkdir and iterates through the processors to find the transition temperature f 
    of a given pixel with a given coordination number and collects the results

    """
    os.chdir(wkdir)
    if wkdir.find('zPBCon') != -1: 
        PBCs = [True,True,True]
    else: 
        PBCs = [True,True,False]
    print(f'Starting {wkdir} with PBCS {PBCs}') 
    
    #Checking for already done: 
    completed = sorted([file for file in os.listdir(wkdir) if file.find('final') != -1 and file.find('Tt') != -1])
    if len(completed) != 3:
        #identify the processor folders

        configs = np.sort(glob.glob('*config*'))
        processors = [file for file in np.sort(glob.glob('*')) if file[0].isnumeric() == True]
        os.chdir(processors[0])
        configs = np.sort(glob.glob('*config*'))

        if os.path.exists('spin_freeze.list'): 
            defects = np.genfromtxt('spin_freeze.list', skip_header = 1)
        else: 
            defects = []

        pos, spin, hc, op = loadAndProcess(configs[0])

        coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)

        coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        #         #activeNNTranslations = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,1]

        coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3
        #         #activeFourSpinTranslations = coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,1]


        startingNNCoordValues = np.unique(coordinationNumberNN)
        startingNNNCoordValues = np.unique(coordinationNumberNNN)
        startingFourSpinCoordValues = np.unique(coordinationNumberFourSpin)

        #print(startingNNNCoordValues, startingFourSpinCoordValues)
         # return to master directory
        os.chdir(wkdir)

        NNCoord = {key: [] for key in startingNNCoordValues}
        NNNCoord = {key: [] for key in startingNNNCoordValues}
        fourSpinCoord = {key: [] for key in startingFourSpinCoordValues}



        # # iterate through the configs and processors
        # has to be for each processor then config

        for j, p in enumerate(processors):
            os.chdir(p)
            print(f'Beginning {p}')
            opTotal = np.zeros(shape = (len(configs), op.shape[0], op.shape[1], op.shape[2], op.shape[3]))
            for i, config in enumerate(configs):
                pos, spin, hc, op = loadAndProcess(config)
                opTotal[i] = op
            transitionTemperatures = np.array([fitPixelTt(opTotal, i,j,k) for i in range(opTotal.shape[2]) for j in range(opTotal.shape[3]) for k in range(opTotal.shape[4])]).reshape((opTotal.shape[2], opTotal.shape[3], opTotal.shape[4], 2)) 

            if os.path.exists('spin_freeze.list'): 
                defects = np.genfromtxt('spin_freeze.list', skip_header = 1)
            else: 
                defects = []



            coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
            coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
            coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
            coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
            coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
            coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   

            for val in np.unique(coordinationNumberNN): 
                mask = coordinationNumberNN == val
                if val in list(NNCoord.keys()): 
                    NNCoord[val] += list(np.ravel(transitionTemperatures[mask]))
                else: 
                    NNCoord.update({val: list(np.ravel(transitionTemperatures[mask]))})
            for val in np.unique(coordinationNumberNNN): 
                mask = coordinationNumberNNN == val
                if val in list(NNNCoord.keys()): 
                    NNNCoord[val] += list(np.ravel(transitionTemperatures[mask]))
                else: 
                    NNNCoord.update({val: list(np.ravel(transitionTemperatures[mask]))})
            for val in np.unique(coordinationNumberFourSpin): 
                mask = coordinationNumberFourSpin == val
                if val in list(fourSpinCoord.keys()): 
                    fourSpinCoord[val] += list(np.ravel(transitionTemperatures[mask]))
                else: 
                    fourSpinCoord.update({val: list(np.ravel(transitionTemperatures[mask]))})
            os.chdir(wkdir)

        for array, filename in zip([NNCoord,NNNCoord, fourSpinCoord], ['CoordNN', 'CoordNNN','fourSpinCoord']): 
                np.save(f'{filename}_Tt_raw.npy', array)
        finalNN = {key: [np.nanmean(NNCoord[key]), np.nanstd(NNCoord[key])/np.sqrt(len(NNCoord[key]))] for key in list(NNCoord.keys())}
        finalNNN = {key: [np.nanmean(NNNCoord[key]), np.nanstd(NNNCoord[key])/np.sqrt(len(NNNCoord[key]))] for key in list(NNNCoord.keys())}
        finalFourSpin= {key: [np.nanmean(fourSpinCoord[key]), np.nanstd(fourSpinCoord[key])/np.sqrt(len(fourSpinCoord[key]))] for key in list(fourSpinCoord.keys())}
        for array, filename in zip([finalNN, finalNNN, finalFourSpin], ['CoordNN', 'CoordNNN','fourSpinCoord']): 
            np.save(f'{filename}_Tt_final.npy', array)


# Identify temps
def loadAndProcessCoordinationNumber(wkdir):
    """
    Takes wkdir and iterates through the processors to find the coordination number each pixel and collects the results

    """
    os.chdir(wkdir)
    if wkdir.find('zPBCon') != -1: 
        PBCs = [True,True,True]
    else: 
        PBCs = [True,True,False]
    print(f'Starting {wkdir} with PBCS {PBCs}') 
    
    #Checking for ones that are already done
    
    completed = sorted([file for file in os.listdir(wkdir) if file.find('final') != -1 and file.find('Tt') == -1])
    configs = sorted([file for file in os.listdir(os.path.join(wkdir, '0')) if file.find('config') != -1])
    completedFS = [file for file in completed if file.find('fourSpin') != -1]
    completedNN = [file for file in completed if file.find('CoordNN_') != -1]
    completedNNN = [file for file in completed if file.find('CoordNNN_') != -1]

    t_fs = []
    t_nn = []
    t_nnn = []
    configTemps = []

    for fs in completedFS: 
        dashes = list(find_all(fs, '_'))
        t_fs.append(fs[dashes[0]+1:dashes[1]])
    for nn in completedNN:
        dashes = list(find_all(nn, '_'))
        t_nn.append(nn[dashes[0]+1:dashes[1]])
    for nnn in completedNNN:
        dashes = list(find_all(nnn, '_'))
        t_nnn.append(nnn[dashes[0]+1:dashes[1]])
    for config in configs:
        dots = list(find_all(config, '.'))
        configTemps.append(config[dots[0]+1:dots[1]])
    
    #toDo = [t for t in configTemps if t not in t_fs or t not in t_nn or t not in t_nnn]
    if len(t_nn) == 0 or len(t_nnn) == 0 or len(t_fs) == 0: 
        configsToDo = configs
    else: 
        configsToDo = [[file for file in configs if file.find(t) != -1][0] for t in configTemps if t not in t_fs or t not in t_nn or t not in t_nnn]
    if len(configsToDo) != 0:
        #identify the processor folders
        configs = np.sort(glob.glob('*config*'))
        #alreadyCalculated = len(np.sort(glob.glob('*fourSpinCoord*')[::2]))
        processors = [file for file in np.sort(glob.glob('*')) if file[0].isnumeric() == True]
        os.chdir(processors[0])
        configs = np.sort(glob.glob('*config*'))

        if os.path.exists('spin_freeze.list'): 
            defects = np.genfromtxt('spin_freeze.list', skip_header = 1)
        else: 
            defects = []

        pos, spin, hc, op = loadAndProcess(configs[0])

        coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)

        coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        #         #activeNNTranslations = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,1]

        coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3
        #         #activeFourSpinTranslations = coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,1]


        startingNNCoordValues = np.unique(coordinationNumberNN)
        startingNNNCoordValues = np.unique(coordinationNumberNNN)
        startingFourSpinCoordValues = np.unique(coordinationNumberFourSpin)

        #print(startingNNNCoordValues, startingFourSpinCoordValues)
         # return to master directory
        os.chdir(wkdir)


        # # iterate through the configs and processors
        for config in configsToDo: 
            print(f'Beginning {config[7:10]} for {wkdir}')
            temp = config[7:10]
            NNCoord = {key: [] for key in startingNNCoordValues}
            NNNCoord = {key: [] for key in startingNNNCoordValues}
            fourSpinCoord = {key: [] for key in startingFourSpinCoordValues}

            for p in processors:
                os.chdir(p)
                pos, spin, hc, op = loadAndProcess(config)
                if os.path.exists('spin_freeze.list'): 
                    defects = np.genfromtxt('spin_freeze.list', skip_header = 1)
                else: 
                    defects = []

                coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
                coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
                coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
                coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
                coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
                coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   

                for val in np.unique(coordinationNumberNN): 
                    mask = coordinationNumberNN == val
                    if val in list(NNCoord.keys()): 
                        NNCoord[val] += list(op[1][mask])
                    else: 
                        NNCoord.update({val: list(op[1][mask])})
                for val in np.unique(coordinationNumberNNN): 
                    mask = coordinationNumberNNN == val
                    if val in list(NNNCoord.keys()): 
                        NNNCoord[val] += list(op[1][mask])
                    else: 
                        NNNCoord.update({val: list(op[1][mask])})
                for val in np.unique(coordinationNumberFourSpin): 
                    mask = coordinationNumberFourSpin == val
                    if val in list(fourSpinCoord.keys()): 
                        fourSpinCoord[val] += list(op[1][mask])
                    else: 
                        fourSpinCoord.update({val: list(op[1][mask])})
                os.chdir(wkdir)
            for array, filename in zip([NNCoord,NNNCoord, fourSpinCoord], ['CoordNN', 'CoordNNN','fourSpinCoord']): 
                np.save(f'{filename}_{temp}_raw.npy', array)
            finalNN = {key: [np.nanmean(NNCoord[key]), np.nanstd(NNCoord[key])/np.sqrt(len(NNCoord[key]))] for key in list(NNCoord.keys())}
            finalNNN = {key: [np.nanmean(NNNCoord[key]), np.nanstd(NNNCoord[key])/np.sqrt(len(NNNCoord[key]))] for key in list(NNNCoord.keys())}
            finalFourSpin= {key: [np.nanmean(fourSpinCoord[key]), np.nanstd(fourSpinCoord[key])/np.sqrt(len(fourSpinCoord[key]))] for key in list(fourSpinCoord.keys())}
            for array, filename in zip([finalNN, finalNNN, finalFourSpin], ['CoordNN', 'CoordNNN','fourSpinCoord']): 
                np.save(f'{filename}_{temp}_final.npy', array)
                

def analyseAndDistribute(master): 
    """Watches a variety of folders and collects finished results
    File system set up is very specific so please ask if help is required to change this 
    """
    destination = master
    processing = [folder for folder in os.listdir(master) if folder.find('Process') != -1][0]
    processingFolder = os.path.join(master, processing)
    
    while True:
        started = []
        active = []
        finished = []
        toRunFolders = [file for file in os.listdir(master) if file.find('toRun') != -1]
        for folder in toRunFolders: 
            wkdir = os.path.join(master, folder)
            toRun = [file for file in sorted(os.listdir(wkdir)) if file.find('.sh') == -1 and file.find('slurm') == -1]
            simLength = {}
            for key in toRun: 
                dashes = list(find_all(key, '_'))
                ks = list(find_all(key, 'K'))
                startTemp = int(key[dashes[1]+1:ks[0]])
                endTemp = int(key[dashes[2]+1:ks[1]])
                step = int(key[dashes[3]+1:ks[2]])
                simLength.update({key: int(abs(startTemp-endTemp)/step + 1)})
            s  = [file for file in toRun if sorted(os.listdir(os.path.join(wkdir, file)))[0].isnumeric()]
            for item in s: 
                started.append(os.path.join(wkdir,item))
            configs = {}
            for file in s: 
                configs.update({file: [f for f in os.listdir(os.path.join(wkdir, file)) if f.find('config') != -1]})
            a = [key for key in list(configs.keys()) if len(configs[key]) != simLength[key]]
            f = [key for key in list(configs.keys()) if len(configs[key]) == simLength[key]]
            for item in a: 
                active.append(os.path.join(wkdir,item))
            for item in f: 
                finished.append(os.path.join(wkdir,item))
    
    
        for file in finished: 
            print(f'{file} has finished, moving to {processingFolder}.')
            slashes = list(find_all(file, '/'))
            filename = file[slashes[-1]+1:]
            shutil.move(file, os.path.join(processingFolder, filename))
    
        for i, folder in enumerate(os.listdir(processingFolder)):
            print(f'{i} of {len(os.listdir(processingFolder))}')
            finished = [file for file in os.listdir(os.path.join(processingFolder, folder)) if file.find('fourSpinCoord') != -1 and file.find('Tt') == -1]
            if len(finished) != 42: 
                print(f'Initiating coordination number for {folder}')
                loadAndProcessCoordinationNumber(os.path.join(processingFolder,folder))
            finished = [file for file in os.listdir(os.path.join(processingFolder, folder)) if file.find('Tt') != -1]
            if len(finished) != 6: 
                transitionTemperatureCoordinationNumber(os.path.join(processingFolder,folder))
    
            finished = [file for file in os.listdir(os.path.join(processingFolder, folder)) if file.find('fourSpinCoord') != -1 and file.find('Tt') == -1]
            coordNo = 0
            tranTemp = 0
            if len(finished) == 42: 
                coordNo = 1
            finished = [file for file in os.listdir(os.path.join(processingFolder, folder)) if file.find('Tt') != -1]
            if len(finished) == 6: 
                tranTemp = 1
            finishedString = 'Finished'
            if coordNo ==1 and tranTemp == 1: 
                us = list(find_all(folder, '_'))
                vacNo = folder[us[-2]+1:folder.find('Vacancy')]
                print(f'Moving {os.path.join(processingFolder, folder)} to {os.path.join(destination, vacNo, finishedString, folder)}')
                shutil.move(os.path.join(processingFolder, folder), os.path.join(destination, vacNo, finishedString, folder))






#autoAverageCalculator(wkdir, 121)