# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:25:58 2022

@author: massey_j
"""

final = pd.DataFrame(recs.domains2individual['300']['fm'])
for t in ['310', '330', '335', '375', '440']:
    final = final.append(recs.domains2individual[t]['fm'])
final['temp'] = final['temp'].astype(np.float)

data = final[final['temp'] == 310]
data = data.append(final[final['temp'] == 335])
data = data.append(final[final['temp'] == 375])
sns.scatterplot(data = data[data['area'] < 2000], x = 'slice', y = 'area', hue = 'temp')

sns.violinplot(data=final, x='temp', y='area')

final = recs.finalIndividualAF[recs.finalIndividualAF['temp'] == 300]
for t in (330, 440):
    final = final.append(recs.finalIndividualAF[recs.finalIndividualAF['temp'] == t])
    
g = sns.relplot(
    data=recs.domains2individual['310']['fm'],
    x="centroid-0", y="centroid-1", hue = 'slice', size="area", sizes=(10, 200),
    palette = cmap
)

g = sns.relplot(
    data=recs.finalIndividualFM[recs.finalIndividualFM['temp'] == 300],
    x="centroid-0", y="centroid-1", hue = 'slice', size="area", sizes=(10, 200),
    palette = cmap
)
g.set(ylim=(40,160), xlim =(25,175))

attrs = ['finalIndividualFM', 'finalIndividualAF']
scans = [[310,335,375], [300, 330, 440]]
labels1 = {'finalIndividualFM': 'fm',
           'finalIndividualAF': 'af'}
labels2 = {310: 'heat',
           300: 'cool'}
finals = {}

for a in attrs:
    for s in scans: 
         recs.generateHeatCoolDataframe(a, s)
         finals.update({'{}_{}'.format(labels1[a], labels2[s[0]]): recs.sortedDF})
         
prob_dists
count the total
count the number in a slice
"""Do this for all andf see what comes out"""
fig, ax = plt.subplots(1,2)
for t in [310, 335, 375]:
    test = finals['af_heat'][finals['af_heat']['temp'] == t]
    #test = test[test['area'] < 100]
    p = []
    a = []
    pe = []
    ae = []
    for s in np.unique(test['slice']):
        temp = test[test['slice'] == s]
        p.append(len(temp))
        a.append(np.sum(temp['area']))
        pe.append(np.sqrt(len(test[test['slice'] == s])))
    pf = p/np.sum(p)
    pe = pf*np.sqrt((np.array(pe)/np.array(p))**2 + (np.sqrt(np.sum(p))/np.sum(p))**2)
    ax[0].errorbar(np.arange(len(p))/len(p), pf, yerr = pe, fmt = 'o')
    ax[1].plot(np.arange(len(p))/len(p), a, 'o', label = t)
fig.legend()

both = []
neither = []
for t,b in zip(test['top'], top['bottom']):
    both.append((t == True)*(b == True))
    neither.append((t == False)*(b == False))   

total = {}    
for t in [310, 335, 375]:
    split = test[test['temp'] == t]
    out = {}
    out = {cat: np.sum(split[cat])/len(split[cat]) for cat in ['both', 'top', 'bottom', 'neither']}
    total.update({t: out})
    
test = finals['fm_heat']
both = []
neither = []
either = []
for t,b in zip(test['top'], test['bottom']):
    both.append((t == True)*(b == True))
    neither.append((t == False)*(b == False))
    either.append((t == False)*(b == True) + (t == True)*(b == False))
test['both'] = both
test['neither'] = neither
test['either'] = either
total = {}    
for t in [310, 335, 375]:
    split = test[test['temp'] == t]
    out = [np.sum(split[cat])/len(split[cat]) for cat in ['top', 'bottom', 'both', 'neither']]
    total.update({t: out})