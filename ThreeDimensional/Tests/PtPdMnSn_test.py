# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:35:10 2022

@author: massey_j
"""

import os
import glob
from pyJM.ThreeDimensional.Lamni import *
from pyJM.BasicFunctions import *


homedir = r'C:\Data\3D Skyrmion\PtPdMnSn\Raw_20220805'
os.chdir(homedir)
files = glob.glob('*')

paramDict =  {'891' : {'H or C': 'H', 
                         'Rot': 0, 
                         'Box': [100, 380, 100, 380], #xx,yy
                         'thresh': 0.0,  
                         'thetaoffset': 0},
              '898' : {'H or C': 'H', 
                       'Rot': 0, 
                      'Box': [100, 380, 100, 380], #xx,yy
                                       'thresh': 0.00,  
                                       'thetaoffset': 0},
             }

#{'898' : {'H or C': 'H', 
#                      'Rot': 0, 
#                      'Box': [150, 450, 100, 350], #xx,yy
#                      'thresh': 0, 
#                      'thetaoffset': 0},
            
#paramDict = None
savePath = r'C:\Data\3D Skyrmion\PtPdMnSn'
recs = Lamni(files[1], homedir, paramDict, 300)
recs.filterAttribute('magProcessed', 6)
#recs.saveParaview(savePath)
# recs.QuiverPlotSingle('x', 100, 10, 10, scale2 = 0.0001, pos = [-.9,1, 0.5, 0.5])
# recs.CalculateVorticity()
# vv = np.sqrt(np.sum(recs.vorticity**2, axis = 0))

import matplotlib.pyplot as plt
import pyvista as pv

# vector_field = recs.filtered/np.sqrt(np.sum(recs.filtered**2, axis = 0))
# mask = circle(vector_field[0,...,0], 125)
# vector_field[:, ~mask, :] = 0
# vector_field = vector_field[:, ::3, ::3, ::3]
# _, nx, ny, nz = vector_field.shape
# size = vector_field[0].size

# origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
# mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)

# mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
# mesh['my'] = mesh['vectors'][:, 2]

# # remove some values for clarity
# num_arrows = mesh['vectors'].shape[0]
# rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
#                              replace=False)

# mesh['vectors'][rand_ints] = 0
# #mesh['scalars'] = mesh['vectors'][:, 2]


# mesh['vectors'][rand_ints] = np.array([0, 0, 0])
# arrows = mesh.glyph(factor=2, geom=pv.Arrow())
# pv.set_plot_theme("document")
# p = pv.Plotter()
# p.add_mesh(arrows, scalars='my', lighting=False, cmap='viridis')
# p.show_grid()
# p.add_bounding_box()

# y_down = [(0, 80, 0),
#           (0, 0, 0),
#           (0, 0, -90)]
# p.show(cpos=y_down)


# test = np.copy(recs.filtered)
# test = test/np.sqrt(np.sum(test**2, axis = 0))
# k = 50
# fig, ax = plt.subplots(1,3)

# mask = circle(test[0,...,0], 125)
# test[:,~mask, :] = np.nan
# for c in (0,1,2): 
#     bar = ax[c].imshow(test[c,...,k], vmin = -1, vmax = 1)
#     ax[c].set_xticks([])
#     ax[c].set_yticks([])
# cax = fig.add_axes([0.925, 0.25, 0.04, 0.5])
# plt.colorbar(bar, cax=cax)


"""Need to implement a way to filter, then save the filter, then perform tasks with it """
"""Do this for the vortivityy to see whats what"""