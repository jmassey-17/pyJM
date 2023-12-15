# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:52:57 2023

@author: massey_j
"""
import threading
import numpy as np
from threading import Thread
import os
import glob
from time import perf_counter

wkdir = r'D:\Data\MC_Backup\CoordinationNumber_10x10x30_20230629\Tests\20230113_10x10x30_400K_300K_5KSteps_50000Steps_RandomStart_0.1Vacancy'

class ProcessorThread(Thread): 
    def __init__(self, wkdir, p, config): 
        Thread.__init__(self)
        self.p = p
        self.wkdir = wkdir
        self.config = config
        
    def run(self): 
        pos, spin, hc, op = loadAndProcess(os.path.join(self.wkdir, self.p, self.config))
        if os.path.exists('vacancy.list'): 
            defects = np.genfromtxt('vacancy.list', skip_header = 1)
        else: 
            defects = []
        start = perf_counter()
        self.coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        self.coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        self.coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        self.coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        self.coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        self.coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   
        print(f'Thread {self.p} executed in {perf_counter()-start} s')
        
class SplitToThread(Thread): 
    def __init__(self, x,y,z, spin, defects, PBCs,p): 
        Thread.__init__(self)
        self.x = x
        self.y = y
        self.z = z
        self.spin = spin
        self.defects = defects
        self.PBCs= PBCs
        self.p = p
    def run(self): 
        start = perf_counter()
        print(f'{self.p}')
        self.coordsNN = np.array([translationChecker(i,j,k, NNTranslations, self.spin, self.PBCs,self.defects) for i, j, k in zip(self.x,self.y,self.z)], dtype = object)
        here1 = perf_counter()
        print(f'part 1 takes {here1-start}s')
        #self.coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        self.coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, self.spin, self.PBCs,self.defects) for i, j, k in zip(self.x, self.y, self.z)], dtype = object)
        here2 = perf_counter()
        print(f'part 2 takes {here2-here1}s')
        #self.coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        self.coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, self.spin, self.PBCs,self.defects) for translationSet in fourSpinTranslations] for i, j, k in zip(self.x, self.y, self.z)], dtype = object)
        #self.coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   
        here3 = perf_counter()
        print(f'part 3 takes {here3-here2}s')
        print(f'Thread {self.p} executed in {perf_counter()-start} s')

def loadAndProcess(file):
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

from scipy.optimize import curve_fit

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
    
def defectChecker(r, defects): 
    temp = False
    for d in defects: 
        if all(r == d): 
            temp = True
    return temp
    
def translationChecker(i,j,k, translationSet, spin, PBCs, defects): 
    toReturn = [any([PBCChecker2(res,size,PBC) == -5 for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spin.shape[1:],PBCs)]) for translation in translationSet]
    result = [[PBCChecker2(res,size,PBC) for res,size,PBC in zip((np.array([i,j,k]) + np.array(translation)), spin.shape[1:],PBCs)] for translation in translationSet]
    defectReturn = [defectChecker(r,defects) for r in result]
    toReturn = np.logical_or(toReturn, defectReturn)
    coordNo = len(translationSet) - sum(toReturn)
    toReturn = [res for res,r in zip(result, toReturn) if r == False]
    while len(toReturn) < len(translationSet): 
        toReturn.append([0,0,0])
    return coordNo, toReturn

NNTranslations, NNNTranslations, fourSpinTranslations = initializeTranslations()

os.chdir(wkdir)
if wkdir.find('zPBCon') != -1: 
    PBCs = [True,True,True]
else: 
    PBCs = [True,True,False]
print(f'Starting {wkdir} with PBCS {PBCs}') 
#identify the processor folders
configs = np.sort(glob.glob('*config*'))
#alreadyCalculated = len(np.sort(glob.glob('*fourSpinCoord*')[::2]))
processors = [file for file in np.sort(glob.glob('*')) if file[0].isnumeric() == True]
os.chdir(processors[0])
configs = np.sort(glob.glob('*config*'))

data = np.genfromtxt(configs[0], skip_header=2)
defaultShape = [int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1]

outs = np.sort([file for file in glob.glob('*out*') if file[5].isnumeric() == True])
defaultShapeOut = np.genfromtxt(outs[0])[:,1:].shape

if os.path.exists('vacancy.list'): 
    defects = np.genfromtxt('vacancy.list', skip_header = 1)
else: 
    defects = []

# pos, spin, hc, op = loadAndProcess(configs[0])

# start = perf_counter()
# coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
# coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)

# coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
# coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
# #         #activeNNTranslations = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,1]

# coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
# coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3
# #         #activeFourSpinTranslations = coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,1]
# print(perf_counter()-start)


# t = []
# start = perf_counter()

# for i in processors: 
#     thread = ProcessorThread(wkdir, i, configs[0])
#     t.append(thread)
    
# for thread in t: 
#     thread.start()

# for thread in t: 
#     thread.join()
# print(f'Threading process takes {perf_counter()-start} s')
# numberOfSplits = 10
start = perf_counter()
t = []
for p in processors:
    pos, spin, hc, op = loadAndProcess(os.path.join(wkdir, p, configs[0]))
    if os.path.exists('vacancy.list'): 
        defects = np.genfromtxt('vacancy.list', skip_header = 1)
    else: 
        defects = []
    indicies = np.arange(spin.shape[1]*spin.shape[2]*spin.shape[3])
    np.random.shuffle(indicies)
    split = 10
    splitSize = spin.shape[1]*spin.shape[2]*spin.shape[3]/split

    for i in range(split): 
        x,y,z = np.unravel_index(indicies[int(i*splitSize): int((i+1)*splitSize)], (spin.shape[1], spin.shape[2], spin.shape[3]))
        # coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i, j, k in zip(x,y,z)], dtype = object)
        thread = SplitToThread(x,y,z, spin, defects, PBCs, i)
        thread.start()
        t.append(thread)
    
for thread in t: 
    thread.join()
print(f'Threading process takes {perf_counter()-start}s for 1 config')

start = perf_counter()
for p in processors:
        os.chdir(os.path.join(wkdir, p))
        pos, spin, hc, op = loadAndProcess(configs[0])
        if os.path.exists('vacancy.list'): 
            defects = np.genfromtxt('vacancy.list', skip_header = 1)
        else: 
            defects = []
        start1 = perf_counter()
        coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        here1 = perf_counter()
        print(f'part 1 takes {here1-start1}s')
        coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        
        coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        here2 = perf_counter()
        print(f'part 2 takes {here2-here1}s')
        coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
        coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
        here3 = perf_counter()
        print(f'part 3 takes {here3-here2}s')
        coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   

print(f'Straight method takes {perf_counter()-start} s for 1 config')
        
        
    
#     coordsNN = np.array([translationChecker(i,j,k, NNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
#     coordinationNumberNN = coordsNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
#     coordsNNN = np.array([translationChecker(i,j,k, NNNTranslations, spin, PBCs,defects) for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
#     coordinationNumberNNN = coordsNNN.reshape(spin.shape[1], spin.shape[2], spin.shape[3],2)[...,0].astype(int)
#     coordsFourSpin = np.array([[translationChecker(i,j,k, translationSet, spin, PBCs,defects) for translationSet in fourSpinTranslations] for i in range(spin.shape[1]) for j in range(spin.shape[2]) for k in range(spin.shape[3])], dtype = object)
#     coordinationNumberFourSpin = np.sum(coordsFourSpin.reshape(spin.shape[1], spin.shape[2], spin.shape[3],len(fourSpinTranslations),2)[...,0].astype(int), axis = -1)/3   
# print(f'Regular process takes {perf_counter()-start} s')

    
           
        