# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:38:46 2023

@author: massey_j
"""

savedir = r'C:\Data\FeRh\Reconstructions_All_20220425\output'
os.chdir(savedir)
from matplotlib import rc
plt.rcParams['font.sans-serif']=['Arial']

plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'mathtext.default':  'regular' })
plt.rcParams["legend.frameon"] = False

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=28)     # fontsize of the axes title
plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

#phi heat
fig, ax = plt.subplots(figsize = (3,6))
temps = ['310', '335', '375']
cmap = plt.cm.CMRmap
datarange = np.arange(len(temps))/len(temps)
colours = cmap(datarange)
#ax.set_xlim([-0.005, 0.005])
#ax.set_ylabel('$z/z_{Max}$')
ax.set_xlabel('$\phi/\phi_{Max}$')
ax.set_ylim([-0.02,1.02])
for key,c in zip(temps, colours):
    x = recs.distributions[key]['fm'][0]
    xe = recs.distributions[key]['fm'][1]
    xe = (x/np.amax(x))*np.sqrt((xe/x)**2 + (xe[np.argmax(x)]/np.amax(x))**2)
    y = np.arange(len(x))/len(x)
    y = y[::-1]
    ax.errorbar(x/np.amax(x), y, xerr = abs(xe), c = c, fmt = 'o',markersize=8)
labels = [temp + 'K' for temp in temps]
ax.set_yticks([0,0.5,1])
fig.legend(labels, bbox_to_anchor=(0, 0, 0.5, 0.92), frameon=True, prop = {'size': 16}, framealpha = 1, edgecolor = 'none')

saveName = 'Exp_Heat_Phi'
fig.savefig('{}.svg'.format(saveName), dpi=1200)

#assymetries heat

fig, ax = plt.subplots(figsize = (3,6))
#ax.set_ylabel('$z/z_{Max}$')
ax.set_xlabel('$A/\phi_{Max}$')
ax.set_ylim([-0.02,1.02])
for key,c in zip(temps, colours):
    x = recs.distributions[key]['fm'][0]
    x2 = (x - x[::-1])/np.amax(x)
    xe = recs.distributions[key]['fm'][1]
    xe2 = np.sqrt(xe**2 + xe[::-1]**2)
    xe3 = x2*np.sqrt((xe2/(x - x[::-1]))**2 + (xe[np.argmax(x)]/np.amax(x))**2)
    y = np.arange(len(x))/len(x)
    y = y[::-1]
    ax.errorbar(x2, y, xerr = abs(xe3), c = c, fmt = 'o', markersize=8)
labels = [temp + 'K' for temp in temps]
ax.set_yticks([0,0.5,1])
#fig.legend(labels, bbox_to_anchor=(0, 0, 1.15, 0.75), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
saveName = 'Exp_Heat_A'
fig.savefig('{}.svg'.format(saveName), dpi=1200)

#phi cool
fig, ax = plt.subplots(figsize = (3.5,6))
temps = ['440', '330', '300']
cmap = plt.cm.viridis
datarange = np.arange(len(temps))/len(temps)
colours = cmap(datarange)
#ax.set_xlim([-0.005, 0.005])
ax.set_ylabel('$z/z_{Max}$')
ax.set_xlabel('$\phi/\phi_{Max}$')
ax.set_ylim([-0.02,1.02])
for key,c in zip(temps, colours):
    x = recs.distributions[key]['fm'][0]
    xe = recs.distributions[key]['fm'][1]
    xe = (x/np.amax(x))*np.sqrt((xe/x)**2 + (xe[np.argmax(x)]/np.amax(x))**2)
    y = np.arange(len(x))/len(x)
    y = y[::-1]
    ax.errorbar(x/np.amax(x), y, xerr = abs(xe), c = c, fmt = 'o', markersize=8)
labels = [temp + 'K' for temp in temps]
ax.set_yticks([0,0.5,1])
fig.legend(labels, bbox_to_anchor=(0, 0, 0.45, 0.45), frameon=True, prop = {'size': 16}, framealpha = 1, edgecolor = 'none')
saveName = 'Exp_Cool_Phi'
fig.savefig('{}.svg'.format(saveName), dpi=1200)

fig, ax = plt.subplots(figsize = (3,6))
#ax.set_ylabel('$z/z_{Max}$')
ax.set_xlabel('$A/\phi_{Max}$')
ax.set_ylim([-0.02,1.02])
for key,c in zip(temps, colours):
    x = recs.distributions[key]['fm'][0]
    x2 = (x - x[::-1])/np.amax(x)
    xe = recs.distributions[key]['fm'][1]
    xe2 = np.sqrt(xe**2 + xe[::-1]**2)
    xe3 = x2*np.sqrt((xe2/(x - x[::-1]))**2 + (xe[np.argmax(x)]/np.amax(x))**2)
    y = np.arange(len(x))/len(x)
    y = y[::-1]
    ax.errorbar(x2, y, xerr = abs(xe3), c = c, fmt = 'o', markersize=8)
labels = [temp + 'K' for temp in temps]
ax.set_yticks([0,0.5,1])
saveName = 'Exp_Cool_A'
fig.savefig('{}.svg'.format(saveName), dpi=1200)

#fig.legend(labels, bbox_to_anchor=(0, 0, 1.15, 0.75), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')

fig, ax = plt.subplots(figsize = (8,6))
ax.set_ylabel('P')
ax.set_xlabel('Surface Attachments')
cats =  ['top', 'bottom', 'both', 'either', 'neither']
for kw,col in zip(['fm_heat', 'af_cool'], ['r','b']):
    recs.generateProbability(kw, cats)
    ax.errorbar(np.arange(len(cats)), recs.probs[1], yerr = recs.probs[2], c = col, fmt = 'o', markersize = 8)
cats2 =  ['', 'Top', 'Bottom', 'Both', 'Either', 'Floating']
ax.set_xticklabels(cats2)
tags = ['FM Domains on Heating', 'AF Domains on Cooling']
fig.legend(tags,bbox_to_anchor=(0, 0, 0.7, 0.85), frameon=True, framealpha = 1, edgecolor = 'none')
saveName = 'P_Python'
fig.savefig('{}.svg'.format(saveName), dpi=1200)
