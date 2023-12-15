# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:34:04 2023

@author: massey_j
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:39:00 2022

@author: massey_j
"""

import numpy as np
from pyJM.ThreeDimensional import Lamni, LamniMulti, LamniOutput
import matplotlib.pyplot as plt
import pyvista as pv
import os
import glob

heatScans = ['310', '335', '375']
coolScans = ['440', '330', '300']

homedir = r'C:\Data\FeRh\Reconstructions_All_20220425\wxmcd'

"""Do a proper plan of how to incorporate the GUI classes and the parameters file for previous measurements """

paramDict = {'300': {'H or C': 'C', 
                      'Rot': -20, 
                      'Box': [75, 215, 40, 215], 
                      'thresh': 0.0001, 
                      'thetaoffset': 0},
              '310': {'H or C': 'H', 
                      'Rot': 20, 
                      'Box': [75, 230, 25, 210], 
                      'thresh': 0.0001, 
                      'thetaoffset': 4},
              '330': {'H or C': 'C', 
                      'Rot': -20, 
                      'Box': [70, 215, 45, 215], 
                      'thresh': 0.0001, 
                      'thetaoffset': 0},
              '335': {'H or C': 'H', 
                      'Rot': 20, 
                      'Box': [60, 220, 35, 210], 
                      'thresh': 0.0001, 
                      'thetaoffset': 5},
              '375': {'H or C': 'H', 
                      'Rot': 23, 
                      'Box': [65, 220, 40, 220], 
                      'thresh': 0.0001, 
                      'thetaoffset': 5},
              '440': {'H or C': 'C', 
                      'Rot': -28, 
                      'Box': [90, 235, 50, 230], 
                      'thresh': 0.0001, 
                      'thetaoffset': 24},
              }     


homedir = r'C:\Data\FeRh\Reconstructions_All_20220425\wxmcd'
os.chdir(homedir)
files = glob.glob('*')
files = [file for file in files if len(file) < 7]
 
# rec = Lamni.Lamni(files[1], homedir, paramDict = paramDict)
# if rec.t == '440': 
#     for i in range(rec.magProcessed.shape[0]):
#         for j in range(rec.magProcessed.shape[-1]): 
#               rec.magProcessed[i, ...,j] = rotate(rec.magProcessed[i, ...,j], 180)
#     #for 440 K only        
#     rec.magProcessed[0] = -rec.magProcessed[0]
#     rec.magProcessed[1]= -rec.magProcessed[1]

# #recs = LamniMulti.LamniMulti(homedir, paramDict = paramDict)
recs = LamniMulti.LamniMulti(homedir, paramDict = paramDict)
# # recs.countDistribution()
recs.domainAnalysis2(thresh = 1)
recs.generateHeatCoolDataframe('finalIndividualFM', [310,335,375])
recs.generateFinalDataframe(['finalFM', 'finalAF'])
recs.generateProbability('fm_heat', ['top', 'bottom', 'both', 'either', 'neither'])
# test = np.zeros_like(rec.magMasks)
# test[rec.magMasks == 1] = 1
# test[rec.magMasks == 0] = -1

# import pyvista as pv
# scalar_field = test
# nx, ny, nz = scalar_field.shape
# size = scalar_field[0].size

# origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
# mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)

# mesh['scalars'] = scalar_field.flatten(order = "F")


# # # remove some values for clarity
# num_arrows = mesh['scalars'].shape[0]
# rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
#                              replace=False)

# pv.set_plot_theme("document")
# p = pv.Plotter()
# p.add_mesh(mesh, scalars=mesh['scalars'], opacity = 0.5, lighting=False, cmap='twilight_shifted')
# p.show_grid()
# p.add_bounding_box()

# y_down = [(0, 80, 0),
#           (0, 0, 0),
#           (0, 0, -90)]
# p.show(cpos=y_down)
# mesh.save('{}.vtk'.format('AF_test'))
