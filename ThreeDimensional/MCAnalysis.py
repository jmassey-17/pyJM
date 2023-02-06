# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:03:22 2023

@author: massey_j
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import glob

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
        if len(test)==0:
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
            
    def calculateAverages(self, wkdir):
        os.chdir(wkdir)

        #identify the processor folders
        processors = [file for file in np.sort(glob.glob('*')) if file[0].isnumeric() == True]

        # Identify temps
        os.chdir(processors[0])
        configs = np.sort(glob.glob('*config*'))

        data = np.genfromtxt(configs[0], skip_header=2)
        defaultShape = [int(max(data[:,0]))+1, int(max(data[:,1]))+1, int(max(data[:,2]))+1]


        # return to master directory
        os.chdir(wkdir)

        # iterate through the configs and processors
        for temp in configs:
            position_av = np.zeros(shape = (3,defaultShape[0], defaultShape[1], defaultShape[2]))
            spin_av = np.zeros(shape = (3,defaultShape[0], defaultShape[1], defaultShape[2]))
            hc_av = np.zeros(shape = (defaultShape[0], defaultShape[1], defaultShape[2]))
            op_av = np.zeros(shape = (2,defaultShape[0], defaultShape[1], defaultShape[2]))
            for p in processors:
                os.chdir(p)
                pos, spin, hc, op = self.loadAndProcess(temp)
                position_av += pos
                spin_av += spin
                hc_av += hc
                op_av += op
                os.chdir(wkdir)
            position_av = position_av/len(processors)
            spin_av = spin_av/len(processors)
            hc_av = hc_av/len(processors)
            op_av = op_av/len(processors)
            final = {'position': position_av, 
                     'spin': spin_av, 
                     'hc': hc_av, 
                     'op': op_av}
            np.save('JM_avtest_{}.npy'.format(temp[7:10]), final)
    
    def loadAverages(self, wkdir):
        os.chdir(wkdir)
        t = []
        processed = np.sort(glob.glob('*av*.npy'))
        pos, spin, hc, op = self.loadProcessedAverage(processed[0])
        spinOut = np.zeros(shape = (len(processed), spin.shape[0], spin.shape[1], spin.shape[2], spin.shape[3]))
        hcOut = np.zeros(shape = (len(processed), hc.shape[0], hc.shape[1], hc.shape[2]))
        opOut = np.zeros(shape = (len(processed), op.shape[0], op.shape[1], op.shape[2], op.shape[3]))
        spinOut[0] = spin
        hcOut[0] = hc
        opOut[0] = op
        for i in range(1, len(processed)):
            pos, spin, hc, op = self.loadProcessedAverage(processed[i])
            spinOut[i] = spin
            hcOut[i] = hc
            opOut[i] = op
        
        for file in processed:
            init = file.find('avtest')
            end = file[init+7:].find('.')
            t.append(int(file[init+7:init+7+end]))
            
        self.t = np.array(t)
        self.spinAv = spinOut
        self.hcAv = hcOut
        self.opAv = opOut 
        print('Averages loaded')
        
    def loadGlobalOutput(self, wkdir):
        os.chdir(wkdir)
        try: 
            files = np.genfromtxt('out_ave.dat', skip_header=1)
            [self.globalT, self.globalJ, self.globalJ1, self.globalJ2, self.globalJ3, 
             self.globalHC, self.globalFMOP, self.globalFMSUS, self.globalAFOP, self.globalAFSUS, self.globalHCAV] = [files[:,i] for i in range(files.shape[-1])]
            print('Global output file loaded successfully')
        except: 
            print(f"Could not find global output file in {os.getcwd}")
        
        
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
        