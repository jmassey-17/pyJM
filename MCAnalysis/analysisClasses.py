# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:19:11 2023

@author: massey_j
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pyJM.BasicFunctions import find_all
from pyJM.MCAnalysis.core import *

class MCFile: 
    def __init__(self, file): 
        self.pos, self.spin, self.hc, self.op = self.loadAndProcess(file)
    
    def viewFile(self): 
        fig, ax = plt.subplots(2,3)
        ax[0,0].imshow(np.swapaxes(np.mean(self.spin[0], axis = 0), 0, 1))
        ax[0,1].imshow(np.swapaxes(np.mean(self.spin[1], axis = 0), 0, 1))
        ax[0,2].imshow(np.swapaxes(np.mean(self.spin[2], axis = 0), 0, 1))
        ax[1,0].imshow(np.swapaxes(np.mean(self.hc, axis = 0), 0, 1))
        ax[1,1].imshow(np.swapaxes(np.mean(self.op[0], axis = 0), 0, 1))
        ax[1,2].imshow(np.swapaxes(np.mean(self.op[1], axis = 0), 0, 1))
        
    def loadAndProcess(self, file):
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

class MCAverage(MCFile):
    def __init__(self, wkdir): 
        os.chdir(wkdir)
        test = glob.glob('*avtest*')
        if wkdir.find('Finished') != -1:
            print(f'Simulations in {wkdir} finished, autofinding average number')
            tempPath = wkdir[wkdir.find('Finished')+9:]
            dashes = list(find_all(tempPath, '_'))
            ks = list(find_all(heatPath, 'K'))
            startTemp = int(tempPath[dashes[1]+1:ks[0]])
            endTemp = int(tempPath[dashes[2]+1:ks[1]])
            step = int(tempPath[dashes[3]+1:ks[2]])
            simNo = int(abs(startTemp-endTemp)/step + 1)
            if len(test)!=simNo:
                print('No averages detected, running average calculator')
                self.calculateAverages(wkdir)
                print('Averages calculated, loading relevant files')
                self.loadAverages(wkdir)
                print('Loading global output')
                self.loadGlobalOutput(wkdir)
            else: 
                print('Averages already calculated, loading in')
                self.loadAverages(wkdir)
                print('Loading global output')
                self.loadGlobalOutput(wkdir)
        else: 
            print('Unable to auto find sim number. Running averages again...')
            self.calculateAverages(wkdir)
            print('Averages calculated, loading relevant files')
            self.loadAverages(wkdir)
            print('Loading global output')
            self.loadGlobalOutput(wkdir)
            
            
    def calculateAverages(self, wkdir):
        os.chdir(wkdir)

        #identify the processor folders
        processors = [file for file in np.sort(glob.glob('*')) if file[0].isnumeric() == True]

        # Identify temps
        os.chdir(processors[0])
        configs = np.sort(glob.glob('*config*'))

        data = np.genfromtxt(configs[0], skip_header=2)
        defaultShape = [int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1]
        
        outs = np.sort([file for file in glob.glob('*out*') if file[5].isnumeric() == True])
        defaultShapeOut = np.genfromtxt(outs[0])[:,1:].shape

        # return to master directory
        os.chdir(wkdir)

        # iterate through the configs and processors
        for config, out in zip(configs,outs): 
            position_av = np.zeros(shape = (3,defaultShape[0], defaultShape[1], defaultShape[2]))
            spin_av = np.zeros(shape = (3,defaultShape[0], defaultShape[1], defaultShape[2]))
            hc_av = np.zeros(shape = (defaultShape[0], defaultShape[1], defaultShape[2]))
            op_av = np.zeros(shape = (2,defaultShape[0], defaultShape[1], defaultShape[2]))
            totalEnergy = np.zeros(shape = defaultShapeOut, dtype = float)
            for p in processors:
                os.chdir(p)
                pos, spin, hc, op = self.loadAndProcess(config)
                totalEnergy += np.genfromtxt(out)[:,1:]
                position_av += pos
                spin_av += spin
                hc_av += hc
                op_av += op
                os.chdir(wkdir)
            position_av = position_av/len(processors)
            spin_av = spin_av/len(processors)
            hc_av = hc_av/len(processors)
            op_av = op_av/len(processors)
            totalEnergy = totalEnergy/len(processors)
            finalEnergy = {'total': totalEnergy[:,0], 
             'J1': totalEnergy[:,1],
             'J2': totalEnergy[:,2],
             'J3': totalEnergy[:,3]}
            final = {'position': position_av, 
                     'spin': spin_av, 
                     'hc': hc_av, 
                     'op': op_av}
            np.save('JM_avtest_{}.npy'.format(config[7:10]), final)
            np.save('JM_energy_{}.npy'.format(out[4:7]), finalEnergy)
    
    def loadAverages(self, wkdir):
        os.chdir(wkdir)
        t = []
        processed = np.sort(glob.glob('*av*.npy'))
        energies = np.sort(glob.glob('*energy*.npy'))
        
        if len(processed) != len(energies): 
            print('Some energies are missing, rerunning the energy calculation.')
            self.calculateAverages(wkdir)
            os.chdir(wkdir)
            t = []
            processed = np.sort(glob.glob('*av*.npy'))
            energies = np.sort(glob.glob('*energy*.npy'))
        
        pos, spin, hc, op = self.loadProcessedAverage(processed[0])
        totalEnergy, J1, J2, J3 = self.loadProcessedEnergy(energies[0])
        
        spinOut = np.zeros(shape = (len(processed), spin.shape[0], spin.shape[1], spin.shape[2], spin.shape[3]))
        hcOut = np.zeros(shape = (len(processed), hc.shape[0], hc.shape[1], hc.shape[2]))
        opOut = np.zeros(shape = (len(processed), op.shape[0], op.shape[1], op.shape[2], op.shape[3]))
            
        totalEnergyOut = np.zeros(shape = (len(processed), totalEnergy.shape[0]))
        J1Out = np.zeros(shape = (len(processed), J1.shape[0]))
        J2Out = np.zeros(shape = (len(processed), J2.shape[0]))
        J3Out = np.zeros(shape = (len(processed), J3.shape[0]))
        
        spinOut[0] = spin
        hcOut[0] = hc
        opOut[0] = op
        totalEnergyOut[0] = totalEnergy
        J1Out[0] = J1
        J2Out[0] = J2
        J3Out[0] = J3
        
        for i in range(1, len(processed)):
            pos, spin, hc, op = self.loadProcessedAverage(processed[i])
            totalEnergy, J1, J2, J3 = self.loadProcessedEnergy(energies[i])
            spinOut[i] = spin
            hcOut[i] = hc
            opOut[i] = op
            totalEnergyOut[i] = totalEnergy
            J1Out[i] = J1
            J2Out[i] = J2
            J3Out[i] = J3
        
        for file in processed:
            init = file.find('avtest')
            end = file[init+7:].find('.')
            t.append(int(file[init+7:init+7+end]))
            
        self.t = np.array(t)
        self.spinAv = spinOut
        self.hcAv = hcOut
        self.opAv = opOut 
        self.totalEnergy = totalEnergyOut
        self.J1 = J1Out
        self.J2 = J2Out
        self.J3 = J3Out
        print('Averages loaded')
        print('Processing coordination numbers')

        files = [file for file in os.listdir(wkdir) if file.find('CoordNN') != -1 and file.find('Tt') == -1]
        final = sorted([file for file in files if file.find('final')!= -1])
        finalNN = sorted([file for file in final if file.find('NNN') == -1])
        finalNNN = sorted([file for file in final if file.find('NNN') != -1])          
        ttFiles = [file for file in os.listdir(wkdir) if file.find('final') != -1 and file.find('Tt') != -1]

        self.NNCoords = {}
        self.NNNCoords = {}
        self.fourSpinCoords = {}
        for file in finalNN: 
            us = list(find_all(file, '_'))
            temp = file[us[0]+1:us[1]]
            t = [[item, np.load(os.path.join(wkdir,file), allow_pickle = True).item().get(item)] for item in list(np.load(os.path.join(wkdir,file), allow_pickle = True).tolist().keys())]
            data = np.array([[row[0], row[1][0], row[1][1]] for row in t])
            self.NNCoords.update({temp: data})
        for file in finalNNN: 
            us = list(find_all(file, '_'))
            temp = file[us[0]+1:us[1]]
            t = [[item, np.load(os.path.join(wkdir,file), allow_pickle = True).item().get(item)] for item in list(np.load(os.path.join(wkdir,file), allow_pickle = True).tolist().keys())]
            data = np.array([[row[0], row[1][0], row[1][1]] for row in t])
            self.NNNCoords.update({temp: data})
        files = [file for file in os.listdir(wkdir) if file.find('fourSpinCoord') != -1]
        final = sorted([file for file in files if file.find('final')!= -1])
        for file in final: 
            us = list(find_all(file, '_'))
            temp = file[us[0]+1:us[1]]
            t = [[item, np.load(os.path.join(wkdir,file), allow_pickle = True).item().get(item)] for item in list(np.load(os.path.join(wkdir,file), allow_pickle = True).tolist().keys())]
            data = np.array([[row[0], row[1][0], row[1][1]] for row in t])
            self.fourSpinCoords.update({temp: data})
        for file in ttFiles: 
            t = [[item, np.load(os.path.join(wkdir,file), allow_pickle = True).item().get(item)] for item in list(np.load(os.path.join(wkdir,file), allow_pickle = True).tolist().keys())]
            data = np.array([[row[0], row[1][0], row[1][1]] for row in t])
            us = list(find_all(file, '_'))
            string = file[:us[-1]]
            setattr(self, string, data)
        raw = [file for file in os.listdir(wkdir) if file.find('Tt_raw') != -1]
        for file in raw:
            t = [[item, np.load(os.path.join(wkdir,file), allow_pickle = True).item().get(item)] for item in list(np.load(os.path.join(wkdir,file), allow_pickle = True).tolist().keys())]
            data = np.array([calcCoordAverage(row) for row in t])
            string = file[:file.find('.')]
            setattr(self, string, data)
        self.calculateMeanEnergies()
        
    def loadGlobalOutput(self, wkdir):
        os.chdir(wkdir)
        try: 
            files = np.genfromtxt('out_ave.dat', skip_header=1)
            [self.globalT, self.globalJ, self.globalJ1, self.globalJ2, self.globalJ3, 
             self.globalHC, self.globalFMOP, self.globalFMSUS, self.globalAFOP, self.globalAFSUS, self.globalHCAV] = [files[:,i] for i in range(files.shape[-1])]
            print('Global output file loaded successfully')
        except: 
            print(f"Could not find global output file in {os.getcwd()}")
        
        
    def calculateRegionalProperties(self, regions):
        """Regions are the z area's that define the regions required 
        example regions = [[0,8],[8,56], [56,64]] will give 3 regions [0-8,8-56,56-64]
        """
        "generate regions"
        
        self.rhc = {f'{region[0]}-{region[1]}': {'av': np.mean(self.hcAv[:,...,region[0]:region[1]], axis = (1,2,3)), 'err': np.std(self.hcAv[:,...,region[0]:region[1]], axis = (1,2,3))/np.sqrt(self.hcAv[:,...,region[0]:region[1]].shape[1]*self.hcAv[:,...,region[0]:region[1]].shape[2]*self.hcAv[:,...,region[0]:region[1]].shape[3])} for region in regions}
        self.rfm = {f'{region[0]}-{region[1]}': {'av': np.mean(self.opAv[:,0,...,region[0]:region[1]], axis = (1,2,3)), 'err': np.std(self.opAv[:,0,...,region[0]:region[1]], axis = (1,2,3))/np.sqrt(self.opAv[:,0,...,region[0]:region[1]].shape[1]*self.opAv[:,0,...,region[0]:region[1]].shape[2]*self.opAv[:,0,...,region[0]:region[1]].shape[3])} for region in regions}
        self.raf = {f'{region[0]}-{region[1]}': {'av': np.mean(self.opAv[:,1,...,region[0]:region[1]], axis = (1,2,3)), 'err': np.std(self.opAv[:,1,...,region[0]:region[1]], axis = (1,2,3))/np.sqrt(self.opAv[:,1,...,region[0]:region[1]].shape[1]*self.opAv[:,1,...,region[0]:region[1]].shape[2]*self.opAv[:,1,...,region[0]:region[1]].shape[3])} for region in regions}
        print(f"Regional properties calculated for {regions}")    
        

    def loadProcessedAverage(self, file):
        data = np.load(file, allow_pickle = True)
        pos = data.item().get('position')
        spin = data.item().get('spin')
        hc = data.item().get('hc')
        op = data.item().get('op')
        return pos, spin, hc, op
    
    def loadProcessedEnergy(self, file):
        return [np.load(file, allow_pickle = True).item().get(item) for item in list(np.load(file, allow_pickle = True).tolist().keys())]


    def viewProcessedTemp(self, temp):
        fig, ax = plt.subplots(2,3)
        ax[0,0].imshow(np.swapaxes(np.mean(self.spinAv[np.argwhere(self.t == temp)[0][0],0], axis = 0), 0, 1))
        ax[0,1].imshow(np.swapaxes(np.mean(self.spinAv[np.argwhere(self.t == temp)[0][0],1], axis = 0), 0, 1))
        ax[0,2].imshow(np.swapaxes(np.mean(self.spinAv[np.argwhere(self.t == temp)[0][0],2], axis = 0), 0, 1))
        ax[1,0].imshow(np.swapaxes(np.mean(self.hcAv[np.argwhere(self.t == temp)[0][0]], axis = 0), 0, 1))
        ax[1,1].imshow(np.swapaxes(np.mean(self.opAv[np.argwhere(self.t == temp)[0][0],0], axis = 0), 0, 1))
        ax[1,2].imshow(np.swapaxes(np.mean(self.opAv[np.argwhere(self.t == temp)[0][0],1], axis = 0), 0, 1))
        
        ax[0,0].set_title('sx')
        ax[0,1].set_title('sy')
        ax[0,2].set_title('sz')
        ax[1,0].set_title('hc')
        ax[1,1].set_title('fm')
        ax[1,2].set_title('af')

        ax[0,0].set_xticks([])
        #ax[0,0].set_yticks([])
        ax[0,1].set_xticks([])
        #ax[0,1].set_yticks([])
        ax[0,2].set_xticks([])
        #ax[0,2].set_yticks([])
        ax[1,0].set_xticks([])
        #ax[1,0].set_yticks([])
        ax[1,1].set_xticks([])
        #ax[1,1].set_yticks([])
        ax[1,2].set_xticks([])
        #ax[1,2].set_yticks([])


        fig.suptitle(f't = {self.t[np.argwhere(self.t == temp)[0][0]]} K', fontsize=16)
    
    def zDepTemp(self):
        out = {}
        """Average Heat Capacity"""
        
        array = getattr(self, 'hcAv')
        new = np.zeros(shape = (array.shape[0], array.shape[-1]))
        err = np.zeros(shape = (array.shape[0], array.shape[-1]))
        for i in range(new.shape[0]):
            for j in range(new.shape[1]):
                new[i,j] = np.mean(array[i,...,j])
                err[i,j] = np.std(array[i,...,j])/np.sqrt(np.sum(abs(array[i,...,j]) > 0))
        out.update({'hcAv': [new, err]})
        
        """Average FM OP"""
        array = getattr(self, 'opAv')[:,0]
        new = np.zeros(shape = (array.shape[0], array.shape[-1]))
        err = np.zeros_like(err)
        for i in range(new.shape[0]):
            for j in range(new.shape[1]):
                new[i,j] = np.mean(array[i,...,j])
                err[i,j] = np.std(array[i,...,j])/np.sqrt(np.sum(abs(array[i,...,j]) > 0))
        out.update({'fm': [new,err]})
        self.zDep = out
        print('z dependence as a function of temperature calculated successfully')
    
    def calculateMeanEnergies(self): 
        self.energies = {energy: [np.mean(getattr(self, energy), axis = 1), np.std(getattr(self, energy), axis = 1)] for energy in ['totalEnergy', 'J1', 'J2', 'J3']}
        
        