# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:07:52 2022

@author: massey_j
"""

"""Number of FM pixels per sheet"""
distributions = {}
for t in list(recs.magMasks.keys()):
    fm = []
    af = []
    fmerr = []
    aferr = []
    for i in range(recs.magMasks[t].shape[-1]): 
        fm.append(np.sum(recs.magMasks[t][...,i] == 1)/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i])))
        fmerr.append(np.sum(recs.magMasks[t][...,i] == 1)/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i]))*np.sqrt((1/np.sum(recs.magMasks[t][...,i] == 1)) + (1/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i])))))
        af.append(np.sum(recs.magMasks[t][...,i] == 0)/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i])))
        aferr.append(np.sum(recs.magMasks[t][...,i] == 0)/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i]))*np.sqrt((1/np.sum(recs.magMasks[t][...,i] == 1)) + (1/np.sum(recs.magDict[t][...,i]>th*np.amax(recs.magDict[t][...,i])))))
    distributions.update({t:{'fm': [np.array(fm), np.array(fmerr)],
                             'af': [np.array(af), np.array(aferr)]}})
    
distributions = {}
for t in list(recs.magMasks.keys()):
    fm = []
    af = []
    fmerr = []
    aferr = []
    for i in range(recs.magMasks[t].shape[-1]): 
        fm.append(np.sum(recs.magMasks[t][...,i] == 1)/np.sum(recs.magMasks[t][...,i]>-1))
        fmerr.append(np.sum(recs.magMasks[t][...,i] == 1)/np.sum(recs.magMasks[t][...,i]>-1)*np.sqrt((1/np.sum(recs.magMasks[t][...,i] == 1)) + (1/np.sum(recs.magMasks[t][...,i]>0))))
        af.append(np.sum(recs.magMasks[t][...,i] == 0)/np.sum(recs.magMasks[t][...,i]>-1))
        aferr.append(np.sum(recs.magMasks[t][...,i] == 0)/np.sum(recs.magMasks[t][...,i]>-1)*np.sqrt((1/np.sum(recs.magMasks[t][...,i] == 1)) + (1/np.sum(recs.magMasks[t][...,i]>0))))
    distributions.update({t:{'fm': [np.array(fm), np.array(fmerr)],
                             'af': [np.array(af), np.array(aferr)]}})

th = 0.01
vol = {t: np.sum(recs.magMasks[t] ==1)/np.sum(recs.magDict[t]>th*np.amax(recs.magDict[t])) for t in list(recs.magMasks.keys())}
volerrs =  {t: np.sum(recs.magMasks[t] == 1)/np.sum(recs.magDict[t]>th*np.amax(recs.magDict[t]))*np.sqrt((1/np.sum(recs.magMasks[t] == 1)) + (1/np.sum(recs.magDict[t]>th*np.amax(recs.magDict[t])))) for t in list(recs.magMasks.keys())}


th = 0.01
vol = {t: np.sum(recs.magMasks[t] ==1)/np.sum(recs.magMasks[t] > -1) for t in list(recs.magMasks.keys())}
volerrs =  {t: np.sum(recs.magMasks[t] == 1)/np.sum(recs.magMasks[t] > -1)*np.sqrt((1/np.sum(recs.magMasks[t] == 1)) + (1/np.sum(recs.magMasks[t] > -1))) for t in list(recs.magMasks.keys())}

fig, ax = plt.subplots()    
for t in ['310','335','375']: 
   u = distributions[t]['fm'][0]/np.sum(distributions[t]['fm'][0])
   e = u*np.sqrt((1/distributions[t]['fm'][0]) + (1/np.sum(distributions[t]['fm'][0])))
   ax.errorbar(np.arange(len(u))/len(u), u, yerr = e, fmt = 'o', label = t)
fig.legend() 
    
"""Both either neither"""
fig, ax = plt.subplots()
for branch in ['af_cool', 'af_heat', 'fm_cool', 'fm_heat']:
    test = finals[branch]
    both = []
    neither = []
    either = []
    for t,b in zip(test['top'], test['bottom']):
        both.append((t == True)*(b == True))
        neither.append((t == False)*(b == False))
        either.append((t == True) + (b==True))
    test['both'] = both
    test['neither'] = neither
    test['either'] = either
    total = {}    
    out = [np.sum(test[cat]> 0)/len(test[cat]) for cat in ['top', 'bottom', 'both', 'neither', 'either']]
    outerr = [np.sum(test[cat]> 0)/len(test[cat])*np.sqrt((1/np.sum(test[cat]> 0)) + (1/len(test[cat]))) for cat in ['top', 'bottom', 'both', 'neither', 'either']]
    total.update({t: [out, outerr]})
    toPlot = total[list(total.keys())[0]]
    ax.errorbar(np.arange(5), toPlot[0], yerr = toPlot[1], fmt = 'o', label = branch)
ax.set_xticks(np.arange(5), ['Top', 'Bottom', 'Both', 'Neither', 'Either'])
fig.legend()


"""Both either neither - temp """
fig, ax = plt.subplots()
branch = 'fm_heat'
for temp in np.unique(finals[branch]['temp']):
    test = finals[branch][finals[branch]['temp'] == temp]
    both = []
    neither = []
    either = []
    for t,b in zip(test['top'], test['bottom']):
        both.append((t == True)*(b == True))
        neither.append((t == False)*(b == False))
        either.append((t == True) + (b==True))
    test['both'] = both
    test['neither'] = neither
    test['either'] = either
    total = {}    
    out = [np.sum(test[cat]> 0)/len(test[cat]) for cat in ['top', 'bottom', 'both', 'neither', 'either']]
    outerr = [np.sum(test[cat]> 0)/len(test[cat])*np.sqrt((1/np.sum(test[cat]> 0)) + (1/len(test[cat]))) for cat in ['top', 'bottom', 'both', 'neither', 'either']]
    total.update({t: [out, outerr]})
    toPlot = total[list(total.keys())[0]]
    ax.errorbar(np.arange(5), toPlot[0], yerr = toPlot[1], fmt = 'o', label = temp)
ax.set_xticks(np.arange(5), ['Top', 'Bottom', 'Both', 'Neither', 'Either'])
ax.set_ylim([-0.1, 1.1])
fig.legend()

"""x,y,z dist"""
n = finals['af_heat']
fig, ax = plt.subplots(1,3)
c = ['r', 'b', 'g']
i = 0
for temp in np.unique(n['temp']):
    n = n[n['temp'] == temp]
    n['x-size'] = abs(n['bbox-3']-n['bbox-0'])
    n['y-size'] = abs(n['bbox-4']-n['bbox-1'])
    n['z-size'] = abs(n['bbox-5']-n['bbox-2'])
    
    n['x-size'] = n['x-size']/np.max(n['x-size'])
    n['y-size'] = n['y-size']/np.max(n['y-size'])
    n['z-size'] = n['z-size']/np.max(n['z-size'])

    x = n[['x-size', 'y-size','z-size']].values 

    ax[0].scatter(x[:,0], x[:,1], c=c[i], label = temp, alpha = 0.5) 
    ax[1].scatter(x[:,0], x[:,2]) 
    ax[2].scatter(x[:,1], x[:,2])
    i+=1
fig.legend()


im = label(recs.magMasks['310'][25:175, 40:160, :])
regions = pd.DataFrame(regionprops_table(im, properties = ('centroid',
                                 'area', 
                                 'bbox', 
                                 'image_filled')))
cleaned = regions[regions['area'] > 10]

cleaned['top'] =  cleaned['bbox-2'] == 0
cleaned['bottom'] = cleaned['bbox-5'] == im.shape[-1]

    

test = np.zeros_like(im)
u = 1
topcount = np.zeros(shape = len(cleaned))
botcount = np.zeros_like(topcount)
for r in cleaned: 
    box = r['bbox']
    test[box[0]:box[3],box[1]:box[4], box[2]:box[-1]] = r['image_filled']*u
    if box[2] == 0: 
        topcount[u-1] = 1
    elif box[-1] == test.shape[-1]: 
        botcount[u-1] = 1
    u+=1

cols = ['bbox-0', 'bbox-1', 'bbox-2', 
        'bbox-3', 'bbox-4', 'bbox-5', 
        'image_filled']
test = np.zeros_like(recs.magMasks['310'][25:175,40:160,:])
u = 1
for x1, y1, z1, x2, y2, z2, im in zip(n['bbox-0'], n['bbox-1'],n['bbox-2'],n['bbox-3'],n['bbox-4'],n['bbox-5'],n['image_filled']): 
    test[x1:x2, y1:y2, z1:z2] = im*u
    u+=1

    
for t in np.unique(finals['fm_heat']['temp']): 
    n = finals['fm_heat'][finals['fm_heat']['temp'] == t]
    print(len(n), np.mean(n['area']), np.mean(n['centroid-0']), np.mean(n['centroid-1']), np.mean(n['centroid-2']))
    
"""Surface/volume charges""" 
fig, ax = plt.subplots(1,2)
k = 10
toShow  = np.arctan2(recs.magProcessed['310'][1,100,k],-recs.magProcessed['310'][0,...,k])
divShow = div[....,k]
ax[0].imshow(toShow, vmin = -np.pi, vmax = np.pi)
ax[1].imshow(divShow)


"""Normalizing the sizes"""
sizes = {}

for t in list(recs.magProcessed.keys()): 
    d = recs.sampleOutline[t] 
    x = len(np.nonzero(d[100,:,0])[0])
    y = len(np.nonzero(d[:,100,0])[0])
    z = d.shape[2]
    vol = np.sum(d)
    sizes.update({t: [6700/x, 8000/y, 145/z, vol]})

for key in list(finals.keys()):
    df = finals[key]
    df['xnew'] = df.apply(lambda row: abs(row['bbox-3']-row['bbox-0'])*sizes['{}'.format(int(row['temp']))][0], axis = 1)
    df['ynew'] = df.apply(lambda row: abs(row['bbox-4']-row['bbox-1'])*sizes['{}'.format(int(row['temp']))][1], axis = 1)
    df['znew'] = df.apply(lambda row: abs(row['bbox-5']-row['bbox-2'])*sizes['{}'.format(int(row['temp']))][2], axis = 1)
    df['volnew'] = df.apply(lambda row: row['area']/sizes['{}'.format(int(row['temp']))][3], axis = 1)

"""To produce the xy graphs 20221124"""
d = recs.finals['fm_heat']
maps = {'xnew': 'x size (nm)',
'ynew': 'y size (nm)',
'znew': 'z size (nm)'} 
d.rename(colums = maps)

d['type'] = 0
d['type'][d['top'] == 1] = 'Top'
d['type'][d['bottom'] == 1] = 'Bottom'
d['type'][d['neither'] == 1] = 'Floating'
d['type'][d['both'] == 1] = 'Both'

d = d.rename(columns = {'type': 'Domain Type'})

sns.scatterplot(data = d, x = 'x size (nm)', y = 'z size (nm)', hue = 'Domain Type', hue_order = ['Top', 'Bottom', 'Floating', 'Both'] )