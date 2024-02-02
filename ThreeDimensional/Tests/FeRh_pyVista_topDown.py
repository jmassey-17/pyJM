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

cmaps = {'arrows': 'coolwarm', 
         'outline': 'twilight_shifted', 
         'magnitude': 'binary',
         'af': 'viridis',
         }

""""""""""""""""""""""""""""""""""""""""""
view = 'FM'
""""""""""""""""""""""""""""""""""""""""""
arrowScale = 4

boxes = {'down': None, 
         'side': [90,110,0,-1], 
         'AF': [36,66,0, 200],
         'FM': [125,155,0,200],
         }

inplaneSkip = 3
outofplaneSkip = 0

colourBy = 'my'

randomRemoval = True

if view == 'down': 
    includeArrows = True
    includeAF = False
    includeOutline = True
    includeScaleBar = False
    includeBoundingBox = True
    includeDirectionalKey = False
    includeMagnetization = True
    includeScaleBarMagnitude = False
    
    tArray = {'440': [0,0],
              '330': [0,1], 
              '300': [0,2],
              '375': [1,2],
              '335': [1,1], 
              '310': [1,0],
              }
    p = pv.Plotter(shape = (2,3), border = False, lighting = 'three lights')
    
elif view == 'side': 
    includeArrows = True
    includeAF = False
    includeOutline = True
    includeScaleBar = False
    includeBoundingBox = True
    includeDirectionalKey = False
    includeMagnetization = True
    includeScaleBarMagnitude = False
    
    tArray = {'440': [0,0],
              '330': [1,0], 
              '300': [2,0],
              '375': [2,1],
              '335': [1,1], 
              '310': [0,1],
              }
    p = pv.Plotter(shape = (3,2), border = False, lighting = 'three lights')

elif view == 'AF': 
    includeArrows = True
    includeAF = False
    includeOutline = True
    includeScaleBar = False
    includeBoundingBox = True
    includeDirectionalKey = False
    includeMagnetization = True
    includeScaleBarMagnitude = False
    t = '330'
    
   
    
elif view == 'FM': 
    includeArrows = True
    includeAF = False
    includeOutline = True
    includeScaleBar = False
    includeBoundingBox = True
    includeDirectionalKey = False
    includeMagnetization = True
    includeScaleBarMagnitude = False
    t = '310'
    
    
  
pv.set_plot_theme("document")
p.background_color = "white"    

views = {'down': (0.0, 0.0, 145), 
        'side': (0,-75,0), 
        'AF': (0.0, 0.0, 125), 
        'FM': (0.0, 0.0, 125)}

zScale = {'down': 1, 
          'side': 5,
          'AF': 1, 
          'FM': 1}

box = boxes[view]

fileName = f'{view}'
if (view == 'down') or (view == 'side'): 
    for i, t in enumerate(list(recs.magProcessed.keys())):
    
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
        offset = [-0.5, -0.5, -0.5]
        
        origin = (-(nx - 1) * 1 / 2 , -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
        mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
        
        mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
        mesh['m'] = mag_field.T.reshape(size)
        
        mesh['mx'] = mesh['vectors'][:, 0]
        mesh['my'] = mesh['vectors'][:, 1]
        mesh['mz'] = mesh['vectors'][:, 2]
        
        #
        
        if randomRemoval: 
            # remove some values for clarity
            num_arrows = mesh['vectors'].shape[0]
            rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                          replace=False)
            mesh['vectors'][rand_ints] = 0
            
        
        colourByDictionary = {
            'angle': {'values': -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1]), 
                      'upperLimit': np.pi, 
                      'lowerLimit': -np.pi},
            'mx': {'values': mesh['vectors'][:, 0], 
                      'upperLimit': 1, 
                      'lowerLimit': -1},
            'my': {'values': mesh['vectors'][:, 1], 
                      'upperLimit': 1, 
                      'lowerLimit': -1},
            'mz': {'values': mesh['vectors'][:, 2], 
                      'upperLimit': 1, 
                      'lowerLimit': -1},
            }
    
        mesh['scalars'] = colourByDictionary[colourBy]['values']
        
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
        if t == '440': 
            test_out = o
            test_af = af
                
        scalar_field = af*1
        scalar_field[af != True] = 0
        scalar_field[af == True] = 250
        
        scalar_field2 = o
        scalar_field3 = self.magDict[t]
        m = np.amax(scalar_field3)
        if box != None:
            scalar_field3 = scalar_field3[box[2]:box[3], box[0]:box[1], :]
                
        if inplaneSkip != 0: 
            scalar_field = scalar_field[::inplaneSkip, ::inplaneSkip, :]
            scalar_field2 = scalar_field2[::inplaneSkip, ::inplaneSkip, :]
            scalar_field3 = scalar_field3[::inplaneSkip, ::inplaneSkip, :]
        if outofplaneSkip != 0: 
            scalar_field = scalar_field[...,::outofplaneSkip]
            scalar_field2 = scalar_field2[...,::outofplaneSkip]
            scalar_field3 = scalar_field3[...,::outofplaneSkip]
        nx, ny, nz = scalar_field.shape
        size = scalar_field[0].size
        
        
        
        origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
        mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
        mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
        mesh3 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
        mesh1['scalars'] = scalar_field.flatten(order = "F")
        mesh2['scalars'] = scalar_field2.flatten(order = "F")
        mesh3['scalars'] = scalar_field3.flatten(order = "F")
    
    
        
        
        p.subplot(tArray[t][0], tArray[t][1])
        opacity = [0,.7]
        if view == 'down': 
            p.camera_position = 'xy'
            p.camera.roll = -90
        p.camera.position = views[view]
        #p.add_mesh(stream.tube(radius=0.1))
        
        if includeArrows:
            p.add_mesh(arrows, scalars='scalars', cmap=cmaps['arrows'], clim = [colourByDictionary[colourBy]['lowerLimit'],colourByDictionary[colourBy]['upperLimit']], show_scalar_bar = False)
            if i == 0: 
                fileName = fileName + '_arrows_'
        if includeScaleBar:
            p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
            if i == 0: 
                fileName = fileName + '_scalebar_'
        if includeAF and (np.sum(af) > 0):
            p.add_volume(mesh1, scalars='scalars', opacity = [0,.8],  cmap=cmaps['af'], show_scalar_bar = False)
            if i == 0: 
                fileName = fileName + '_AF_'
        if includeOutline:
            p.add_volume(mesh2, scalars='scalars', cmap=cmaps['outline'], opacity = opacity, show_scalar_bar = False)
            if i == 0: 
                fileName = fileName + '_outline_'
        if includeBoundingBox:
            p.add_bounding_box()
            if i == 0: 
               fileName = fileName + '_boundingBox_'
        if includeDirectionalKey: 
            p.add_axes(labels_off = True, line_width = 5)
            if i == 0: 
                fileName = fileName + '_key_'
        if includeMagnetization:
            p.add_volume(mesh3, scalars='scalars', cmap=cmaps['magnitude'], clim = [m/6, m/4], opacity = [0,.1], show_scalar_bar = False)
            # [np.amax(mesh3['scalars'])/6, np.amax(mesh3['scalars'])/3]for down
            if i == 0: 
                fileName = fileName + '_magnitude_'
        if includeScaleBarMagnitude:
            p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
            if i == 0: 
                fileName = fileName + '_mBar_'
    # p.save_graphic(f'{fileName}.svg')
    
    p.show()
else: 

    if box == None:
        f = getattr(self, field)[t]
    else: 
        f = getattr(self, field)[t][:, box[2]:box[3], box[0]:box[1], :]
             
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
    
    
    origin = (-(nx - 1) * 1 / 2 , -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
    mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
    
    mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
    mesh['m'] = mag_field.T.reshape(size)
    
    mesh['mx'] = mesh['vectors'][:, 0]
    mesh['my'] = mesh['vectors'][:, 1]
    mesh['mz'] = mesh['vectors'][:, 2]
    
    #
    
    if randomRemoval: 
        # remove some values for clarity
        num_arrows = mesh['vectors'].shape[0]
        rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
                                      replace=False)
        mesh['vectors'][rand_ints] = 0
        
    
    colourByDictionary = {
        'angle': {'values': -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1]), 
                  'upperLimit': np.pi, 
                  'lowerLimit': -np.pi},
        'mx': {'values': mesh['vectors'][:, 0], 
                  'upperLimit': 1, 
                  'lowerLimit': -1},
        'my': {'values': mesh['vectors'][:, 1], 
                  'upperLimit': 1, 
                  'lowerLimit': -1},
        'mz': {'values': mesh['vectors'][:, 2], 
                  'upperLimit': 1, 
                  'lowerLimit': -1},
        }

    mesh['scalars'] = colourByDictionary[colourBy]['values']
    
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
    scalar_field3 = self.magDict[t]
    m = np.amax(scalar_field3)
    if box != None:
        scalar_field3 = scalar_field3[box[2]:box[3], box[0]:box[1], :]
            
    if inplaneSkip != 0: 
        scalar_field = scalar_field[::inplaneSkip, ::inplaneSkip, :]
        scalar_field2 = scalar_field2[::inplaneSkip, ::inplaneSkip, :]
        scalar_field3 = scalar_field3[::inplaneSkip, ::inplaneSkip, :]
    if outofplaneSkip != 0: 
        scalar_field = scalar_field[...,::outofplaneSkip]
        scalar_field2 = scalar_field2[...,::outofplaneSkip]
        scalar_field3 = scalar_field3[...,::outofplaneSkip]
    nx, ny, nz = scalar_field.shape
    size = scalar_field[0].size
    
    
    
    origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
    mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
    mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
    mesh3 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.*zScale[view]), origin)
    mesh1['scalars'] = scalar_field.flatten(order = "F")
    mesh2['scalars'] = scalar_field2.flatten(order = "F")
    mesh3['scalars'] = scalar_field3.flatten(order = "F")

    p = pv.Plotter()
    pv.set_plot_theme("document")
    p.camera_position = 'xy'
    p.camera.roll = -90
    p.camera.position = views[view]   
    
    opacity = [0,.7]
    
    p.camera.position = views[view]

    if includeArrows:
        p.add_mesh(arrows, scalars='scalars', cmap=cmaps['arrows'], clim = [colourByDictionary[colourBy]['lowerLimit'],colourByDictionary[colourBy]['upperLimit']], show_scalar_bar = False)
        fileName = fileName + '_arrows_'
    if includeScaleBar:
        p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
        fileName = fileName + '_scalebar_'
    if includeAF and (np.sum(af) > 0):
        p.add_volume(mesh1, scalars='scalars', opacity = [0,.8],  cmap=cmaps['af'], show_scalar_bar = False)
        fileName = fileName + '_AF_'
    if includeOutline:
        p.add_volume(mesh2, scalars='scalars', cmap=cmaps['outline'], opacity = opacity, show_scalar_bar = False)
        fileName = fileName + '_outline_'
    if includeBoundingBox:
        p.add_bounding_box()
        fileName = fileName + '_boundingBox_'
    if includeDirectionalKey: 
        p.add_axes(labels_off = True, line_width = 5)
        fileName = fileName + '_key_'
    if includeMagnetization:
        p.add_volume(mesh3, scalars='scalars', cmap=cmaps['magnitude'], clim = [m/6, m/4], opacity = [0,.1], show_scalar_bar = False)
        # [np.amax(mesh3['scalars'])/6, np.amax(mesh3['scalars'])/3]for down
        fileName = fileName + '_magnitude_'
    if includeScaleBarMagnitude:
        p.add_scalar_bar(title = '',n_labels = 0, n_colors = 512)
        fileName = fileName + '_mBar_'
# p.save_graphic(f'{fileName}.svg')

    p.show()
