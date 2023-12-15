# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:56:13 2023

@author: massey_j
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import shutil
import pyvista as pv


file = r'C:\Data\3D AF\Diamond Sept 23\1013319_Averages\modes.h5'
data = h5.File(file)
averageImage = np.array(data['entry_1/image_2/data'])


m = abs(averageImage) > 0.1*np.amax(abs(averageImage))
amp = abs(averageImage)*m
phase = np.arctan2(averageImage.imag, averageImage.real)*m
phase[~m] = np.nan

#box = [135, 165, 200, 300, 200, 300]
scalar_field = phase
nx, ny, nz = scalar_field.shape
size = scalar_field[0].size


origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)

mesh['scalars'] = scalar_field.flatten(order = "F")
#mesh2 = mesh.scale([10.0, 1, 1], inplace=False)

pv.set_plot_theme("document")
p = pv.Plotter()
p.camera_position = 'xy'
p.camera.roll = -90
p.scale = [3,2,1]
# # #p.camera.azimuth = 0
# p.camera.elevation = -45

p.camera.position =  (0.0, 0.0, 90)

#stream, src = mesh.streamlines(vectors = 'vectors', return_source=True)

#p.camera.position = (38.72378309369189, 0.0, 38.72378309369189)

opacity = [0,.7]
#p.add_mesh(stream.tube(radius=0.1))
p.add_volume(mesh, scalars='scalars', cmap='viridis', clim = [-np.pi, np.pi], show_scalar_bar = False)
#p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
# p.add_bounding_box()
p.add_axes(labels_off = True, line_width = 5)


y_down = [(0, 80, 0),
          (0, 0, 0),
          (0, 0, -90)]

z_down = [(0, 0, 180),
(0, 0, 0),
(-40, 1, 0)]
# p.save_graphic(f'bar_topDown.svg')
p.show()
