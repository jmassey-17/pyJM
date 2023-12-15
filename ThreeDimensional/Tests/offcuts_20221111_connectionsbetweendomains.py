# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:31:44 2022

@author: massey_j
"""

r =regions[3]
attrs = [a for a in r.__dir__() if not a.startswith('_')]
print(attrs)

image = recs.magMasks['310'][25:175,40:160,:]
test = np.zeros_like(image)
for i in range(image.shape[-1]):
    label_img = label(image[...,i])
    regions = regionprops(label_img)
    #plt.imshow(image)
    u = 1
    for r in regions: 
        #plt.plot(r['centroid'][1], r['centroid'][0], 'x')
        box = r['bbox']
        test[box[0]:box[2],box[1]:box[3], i] = r['image_filled']*u
        u+=1


connected = np.zeros(shape = (np.amax(test).astype(int), np.amax(test).astype(int), test.shape[-1]-1))  
for i in range(connected.shape[-1]): 
    for j in range(1,np.amax(test[...,i]).astype(int)): #original layer
        for k in range(1,np.amax(test[...,i+1]).astype(int)): #layer after
            coords_a = test[...,i] == j
            coords_b = test[...,i+1] == k
            connected[j,k,i] = np.sum(coords_a*coords_b) > 0
            
"""This goes through each domain on one layer and finds the domains in the other layers that overlap"""
"""Any that dont overlap through all layers are not continuous"""
"""Needs extending to all layers"""
connected2 = np.zeros(shape = (np.amax(test[...,0]).astype(int), np.amax(test).astype(int), test.shape[-1]))  
for j in range(1,np.amax(test[...,0]).astype(int)): #original layer
    for i in range(1,connected2.shape[-1]):     
        for k in range(1,np.amax(test[...,i]).astype(int)): #layer after
            coords_a = test[...,0] == j
            coords_b = test[...,i] == k
            connected2[j,k,i] = np.sum(coords_a*coords_b) > 0
            
arrays = {}
for i in range(1,np.amax(test[...,0]).astype(int)): 
    here = np.zeros_like(test)
    here[test[...,0] == i, 0] = i
    mask = connected2[i] == True
    for j in range(mask.shape[0]): 
        for k in range(mask.shape[1]):
            if mask[j,k] == True:
                here[test[...,k] == j, k] = i
    arrays.update({i:here})
    
    # """Check this on monday"""
    # "select the temp"
    # self.intermediate = {}
    # self.finalIndividualFM['connections'] = 0
    # for t in list(self.magMasks.keys()): 
    #     df = self.finalIndividualFM[self.finalIndividualFM['temp'] == np.int(t)]
    #     test = np.zeros(shape = (150, 120, len(np.unique(df['slice']))))
    #     s = df['slice']/np.unique(df['slice'])[1]
    #     box = df['bbox']
    #     image = df['image_filled']
    #     identifier = df['domainIdentifier']
    #     for i in range(test.shape[-1]):
    #         here = np.where(s == i)
    #         temps = s[here]
    #         tempb = box[here]
    #         tempim = image[here]
    #         tempid = identifier[here]
    #         for j in range(temps.shape[-1]):  
    #             test[tempb[j][0]:tempb[j][2],tempb[j][1]:tempb[j][3], i] = tempim[j]*tempid[j]
        
    #     self.intermediate.update({t: test})
        
    #     "Run for all slices and feedback into the df"""
    #     connecteds = {}
    #     for i in np.unique(s): 
    #         connected2 = np.zeros(shape = (np.amax(test[...,i]).astype(int), np.amax(test).astype(int), test.shape[-1]))  
    #         for j in range(1,np.amax(test[...,0]).astype(int)): #original layer
    #             for i in range(1,connected2.shape[-1]):     
    #                 for k in range(1,np.amax(test[...,i]).astype(int)): #layer after
    #                     coords_a = test[...,0] == j
    #                     coords_b = test[...,i] == k
    #                     connected2[j,k,i] = np.sum(coords_a*coords_b) > 0
    #         connecteds.update({i: connected2})
        
    #     for i in np.unique(s): 
    #         for ident in identifier:
    #             self.finalIndividualFM['connections'][self.finalIndividualFM['slice'] == s and self.finalIndividual['domainIdentifier'] == ident] = connecteds[s][ident] 
        
        



# = regions[regions['area'] > thresh]