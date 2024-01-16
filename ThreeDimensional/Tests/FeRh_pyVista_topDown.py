# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:56:13 2023

@author: massey_j
"""
import os
wkdir = r'C:\Data\FeRh\Figures_20240104'
if os.path.exists(wkdir) != True: 
    os.mkdir(wkdir)
    os.chdir(wkdir)
else: 
    os.chdir(wkdir)

self = recs
field = 'magProcessed'
t = '330'
cmap = 'twilight_shifted'
view = 'down'
i = 0
arrowScale = 3
pv.set_plot_theme("paraview")
p = pv.Plotter(shape = (2,3), border = False)
p.background_color = "white"
# p.camera_position = 'xy'
# p.camera.roll = -90
box = None
inplaneSkip = 5
outofplaneSkip = 0
for t in list(recs.zoomedFinal.keys()):
    # t = '440'
    # i += 1
    # if i == 2: 
    #     break
    
    
    if t == None: 
        if box == None:
            f = getattr(self, field)
        else: 
            f = getattr(self, field)[:, box[2]:box[3], box[0]:box[1], :]
    else:  
        if box == None:
            f = getattr(self, field)[t]
        else: 
            f = getattr(self, field)[t][:, box[2]:box[3], box[0]:box[1], :]
            
    #f = self.vorticity[t]['raw']
    
    vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
    mag_field = np.sqrt(np.sum(f**2, axis = 0))/np.nanmax(np.sqrt(np.sum(f**2, axis = 0)))
    vector_field[np.isnan(vector_field)] = 0
    mag_field[np.isnan(mag_field)] = 0
    if inplaneSkip != 0: 
        vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
        mag_field = mag_field[::inplaneSkip, ::inplaneSkip, :]
    if outofplaneSkip != 0: 
        vector_field = vector_field[:,..., ::outofplaneSkip]
        mag_field = mag_field[..., ::outofplaneSkip]

    _, nx, ny, nz = vector_field.shape
    size = vector_field[0].size
    
    origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
    mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
    
    mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
    mesh['m'] = mag_field.T.reshape(size)
    
    mesh['mx'] = mesh['vectors'][:, 0]
    mesh['my'] = mesh['vectors'][:, 1]
    mesh['mz'] = mesh['vectors'][:, 2]
    
    # # remove some values for clarity
    # num_arrows = mesh['vectors'].shape[0]
    # rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
    #                              replace=False)
    
    #mesh['vectors'][rand_ints] = 0
    mesh['scalars'] = -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1])
    #mesh['scalars'] = mesh['vectors'][:,2]
    
    
    #mesh['vectors'][rand_ints] = np.array([0, 0, 0])
    arrows = mesh.glyph(factor=arrowScale, geom=pv.Arrow())
    sf = 'magMasks'
    
    if t == None: 
        afAndOutline = 1 - getattr(self, sf)
        o = 1- getattr(self, 'sampleOutline')
        af = afAndOutline - (o)
        if box != None: 
            o = o[box[2]:box[3], box[0]:box[1], :]
            af = af[box[2]:box[3], box[0]:box[1], :]
    else:  
        afAndOutline = 1 - getattr(self, sf)[t]
        o = 1 - getattr(self, 'sampleOutline')[t]
        af = afAndOutline - (o)
        if box != None: 
            o = o[box[2]:box[3], box[0]:box[1], :]
            af = af[box[2]:box[3], box[0]:box[1], :]
            
    scalar_field = af*1
    scalar_field[af != True] = 0
    scalar_field[af == True] = 250
    
    scalar_field2 = o
            
    if inplaneSkip != 0: 
        scalar_field = scalar_field[::inplaneSkip, ::inplaneSkip, :]
        scalar_field2 = scalar_field2[::inplaneSkip, ::inplaneSkip, :]
    if outofplaneSkip != 0: 
        scalar_field = scalar_field[...,::outofplaneSkip]
        scalar_field2 = scalar_field2[...,::outofplaneSkip]
    nx, ny, nz = scalar_field.shape
    size = scalar_field[0].size
    
    
    origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
    mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
    mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
    mesh1['scalars'] = scalar_field.flatten(order = "F")
    mesh2['scalars'] = scalar_field2.flatten(order = "F")
    #mesh1['scalars'][rand_ints] = 0
    
    # # remove some values for clarity
    #num_arrows = mesh1['scalars'].shape[0]
    #rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
    #                             replace=False)
    
    
    # pv.set_plot_theme("document")
    # p = pv.Plotter()
    # if view == 'down':
    #     p.camera_position = 'xy'
    #     p.camera.roll = -90
        # # #p.camera.azimuth = 0
        # p.camera.elevation = -45
    #     p.camera.position =  (0.0, 0.0, 90)
    # else: 
    #     p.camera_position = 'xy'
    #     p.camera.roll = -90
    #     #p.camera.azimuth = -30
    #     p.camera.elevation = -45
    #     p.camera.position = (56.69549082747431, 0.0, 56.69549082747431)
    tArray = {'440': [0,0],
              '330': [0,1], 
              '300': [0,2],
              '375': [1,0],
              '335': [1,1], 
              '310': [1,2],
              }
    p.subplot(tArray[t][0], tArray[t][1])
    opacity = [0,.7]
    #p.add_mesh(stream.tube(radius=0.1))
    p.add_mesh(arrows, scalars='scalars', lighting=False, cmap=cmap, clim = [-np.pi, np.pi], show_scalar_bar = False)
    #p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
   # p.add_volume(mesh1, scalars='scalars', opacity = opacity, show_scalar_bar = False)
    p.add_volume(mesh2, scalars='scalars', cmap=cmap, opacity = opacity, show_scalar_bar = False)
    p.camera_position = 'xy'
    p.camera.roll = -90
    p.camera.position =  (0.0, 0.0, 90)
    p.add_bounding_box()
    #p.add_axes(labels_off = True, line_width = 5)
    
    
    y_down = [(0, 80, 0),
              (0, 0, 0),
              (0, 0, -90)]
    
    z_down = [(0, 0, 180),
    (0, 0, 0),
    (-40, 1, 0)]
p.save_graphic(f'{t}_topDown_IPSkip_{inplaneSkip}_view_{view}_all.svg')
p.show()
