# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:26:10 2023

@author: massey_j
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:56:13 2023

@author: massey_j
"""
import os
wkdir = r'C:\Data\FeRh\Figures_20230912\FMDomain'
if os.path.exists(wkdir) != True: 
    os.mkdir(wkdir)
    os.chdir(wkdir)
else: 
    os.chdir(wkdir)
#for t in list(recs.zoomedFinal.keys()):

self = recs
field = 'zoomedFinal'
#Fm
t = '310'
box = None
box2 = [80,120,20,50]

# #AF 
# t = '330'
# box2 = [60, 120, 40, 100]
# inplaneSkip = 5
# outofplaneSkip = 3

#box2 = [val//inplaneSkip for val in box2]





sf = 'zoomedMasks'

if t == None: 
    if box == None:
        f = getattr(self, field)
    else: 
        f = getattr(self, field)
        b = np.zeros(shape = f.shape[1:])
        b[box[2]:box[3], box[0]:box[1], :] = 1
        f = f[:, box[2]:box[3], box[0]:box[1], :]
else:  
    if box == None:
        f = getattr(self, field)[t]
        b = np.zeros(shape = f.shape[1:])
        b[box2[2]:box2[3], box2[0]:box2[1], :] = 1
    else: 
        f = getattr(self, field)[t]
        
        
#f = self.vorticity[t]['raw']

vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
mag_field = np.sqrt(np.sum(f**2, axis = 0))/np.nanmax(np.sqrt(np.sum(f**2, axis = 0)))
vector_field[np.isnan(vector_field)] = 0
mag_field[np.isnan(mag_field)] = 0
if inplaneSkip != 0: 
    vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
    mag_field = mag_field[::inplaneSkip, ::inplaneSkip, :]
    b = b[::inplaneSkip, ::inplaneSkip, :]
if outofplaneSkip != 0: 
    vector_field = vector_field[:,..., ::outofplaneSkip]
    mag_field = mag_field[..., ::outofplaneSkip]
    b = b[..., ::outofplaneSkip]

_, nx, ny, nz = vector_field.shape
size = vector_field[0].size

origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
mesh = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)

mesh['vectors'] = vector_field[0:3].T.reshape(size, 3)
mesh['m'] = mag_field.T.reshape(size)

mesh['mz'] = mesh['vectors'][:, 2]

# # remove some values for clarity
#num_arrows = mesh['vectors'].shape[0]
#rand_ints = np.random.choice(num_arrows - 1, size=int(num_arrows - 2*num_arrows / np.log(num_arrows + 1)),
  #                           replace=False)

#mesh['vectors'][rand_ints] = 0
mesh['scalars'] = -np.arctan2(mesh['vectors'][:, 0], mesh['vectors'][:,1])


#mesh['vectors'][rand_ints] = np.array([0, 0, 0])
arrows = mesh.glyph(factor=2, geom=pv.Arrow())

s = 1 - getattr(self, sf)[t]
af = (getattr(self, sf)[t] < 0.1) & (getattr(self, sf)[t] > -0.5)



if t == None: 
    if box == None:
        s = 1 - getattr(self, sf)
        af = (getattr(self, sf) < 0.1) & (getattr(self, sf) > -0.5)
        o = (getattr(self, sf) < -0.9)
    else: 
        s = 1 - getattr(self, sf)[box[2]:box[3], box[0]:box[1], :]
        af = (getattr(self, sf) < 0.1) & (getattr(self, sf) > -0.5)
        o = (getattr(self, sf) < -0.9)[box[2]:box[3], box[0]:box[1], :]
        af = af[box[2]:box[3], box[0]:box[1], :]
else:  
    if box == None:
        s = 1 - getattr(self, sf)[t]
        af = (getattr(self, sf)[t] < 0.1) & (getattr(self, sf)[t] > -0.5)
        o = (getattr(self, sf)[t] < -0.9)
    else: 
        s = 1 - getattr(self, sf)[t][box[2]:box[3], box[0]:box[1], :]
        af = (getattr(self, sf)[t] < 0.1) & (getattr(self, sf)[t] > -0.5)
        o = (getattr(self, sf)[t] < -0.9)
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

scalar_field = b
origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2)
mesh1 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
mesh2 = pv.UniformGrid((nx, ny, nz), (1., 1., 1.), origin)
mesh1['scalars'] = scalar_field.flatten(order = "F")
mesh2['scalars'] = scalar_field2.flatten(order = "F")


pv.set_plot_theme("document")
p = pv.Plotter()
p.camera_position = 'xy'
p.camera.roll = -90
# # #p.camera.azimuth = 0
# p.camera.elevation = -45

p.camera.position =  (0.0, 0.0, 90)

opacity = [0,.7]
opacity2 = [0,.5]
p.add_mesh(arrows, scalars='scalars', lighting=False, cmap='twilight_shifted', clim = [-np.pi, np.pi], show_scalar_bar=False)
p.add_volume(mesh1, opacity = opacity2, cmap = 'Greens', show_scalar_bar = False)
p.add_volume(mesh2, scalars='scalars', cmap='twilight_shifted', opacity = opacity, show_scalar_bar = False)
p.add_bounding_box()
#p.add_axes(labels_off = True, line_width = 5)


#p.save_graphic(f'{t}_topdown_withgreybox.svg')
p.show()
