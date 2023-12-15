# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:56:13 2023

@author: massey_j
"""
import os
wkdir = r'C:\Data\FeRh\Figures_20230912\sideView_20230918'
if os.path.exists(wkdir) != True: 
    os.mkdir(wkdir)
    os.chdir(wkdir)
else: 
    os.chdir(wkdir)
#for t in list(recs.zoomedFinal.keys()):
self = recs
field = 'zoomedFinal'
#for t in list(recs.zoomedFinal.keys()):
t = '335'
box = [90, 110,70,130]
inplaneSkip = 2
outofplaneSkip = 2




sf = 'zoomedMasks'

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
    
# # for the cross-tie wall
# test = np.zeros_like(recs.zoomedFinal['440'])
# for i in range(test.shape[0]): 
#     for j in range(test.shape[-1]): 
#         test[i,...,j] = rotate(recs.zoomedFinal['440'][i,...,j], 45)
# f = test[:,25:150,75:125,:]

# vector_field = f/np.sqrt(np.sum(f**2, axis = 0))
# if inplaneSkip != 0: 
#     vector_field = vector_field[:, ::inplaneSkip, ::inplaneSkip, :]
# if outofplaneSkip != 0: 
#     vector_field = vector_field[:,..., ::outofplaneSkip]
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


pv.set_plot_theme("document")
p = pv.Plotter()
p.camera_position = 'xz'
#p.camera.roll = -90
# # # #p.camera.azimuth = 0
#p.camera.elevation = -45



#p.camera.position = (56.69549082747431, 0.0, 56.69549082747431)


p.camera.position =  (0,-97.01723378487242, 0.0)

opacity = [0,.7]
p.add_mesh(arrows, scalars='scalars', lighting=False, cmap='twilight_shifted', clim = [-np.pi, np.pi], show_scalar_bar=False)
p.add_volume(mesh1, scalars='scalars', opacity = opacity, show_scalar_bar = False)
#p.add_volume(mesh2, scalars='scalars', cmap='twilight_shifted', opacity = opacity, show_scalar_bar = False)
p.add_bounding_box()
p.add_axes(labels_off = True, line_width = 5)


y_down = [(0, 80, 0),
          (0, 0, 0),
          (0, 0, -90)]

z_down = [(0, 0, 180),
(0, 0, 0),
(-40, 1, 0)]
#p.save_graphic(f'axis_sideview.svg')
p.show()
