# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:38:46 2023

@author: massey_j
"""
from pyJM.BasicFunctions import dateToSave

savedir = r'C:\Data\FeRh\Reconstructions_All_20220425\output'
os.chdir(savedir)
from matplotlib import rc
plt.rcParams['font.sans-serif']=['Arial']

plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'mathtext.default':  'regular' })
plt.rcParams["legend.frameon"] = False

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=28)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title


# fig, ax = plt.subplots(1, 2, figsize = (8,9), sharey = True)

# #should go phi cool, A cool, phi heat, A heat

# #phi cool
# temps = ['440', '330', '300']
# cmap = plt.cm.viridis
# datarange = np.arange(len(temps))/len(temps)
# colours = cmap(datarange)
# #ax.set_xlim([-0.005, 0.005])
# ax[0].set_ylabel('$z/z_{Max}$')
# ax[0].set_xlabel(r'$\langle \phi\rangle_{x,y}/\langle \phi\rangle_{x,y}^{Max}$')
# ax[0].set_ylim([-0.02,1.02])
# ax[0].set_xlim([0.7, 1.05])
# for key,c in zip(temps, colours):
#     x = recs.distributions[key]['fm'][0]
#     xe = recs.distributions[key]['fm'][1]
#     xe = (x/np.amax(x))*np.sqrt((xe/x)**2 + (xe[np.argmax(x)]/np.amax(x))**2)
#     y = np.arange(len(x))/len(x)
#     y = y[::-1]
#     ax[0].errorbar(x/np.amax(x), y, xerr = abs(xe), c = c, fmt = 'o', markersize=12)
# labels = [temp + ' K' for temp in temps]
# ax[0].set_yticks([0,0.5,1])
# ax[0].legend(labels, bbox_to_anchor=(0, 0, 0.6, 0.25), frameon=False, framealpha = 1, edgecolor = 'none', title = 'T') #prop = {'size': 16}

# #phi heat
# temps = ['310', '335', '375']
# cmap = plt.cm.CMRmap
# datarange = np.arange(len(temps))/len(temps)
# colours = cmap(datarange)
# ax[1].set_xlabel(r'$\langle \phi\rangle_{x,y}/\langle \phi\rangle_{x,y}^{Max}$')
# ax[1].set_ylim([-0.02,1.02])
# ax[1].set_xlim([0.7, 1.05])
# for key,c in zip(temps, colours):
#     x = recs.distributions[key]['fm'][0]
#     xe = recs.distributions[key]['fm'][1]
#     xe = (x/np.amax(x))*np.sqrt((xe/x)**2 + (xe[np.argmax(x)]/np.amax(x))**2)
#     y = np.arange(len(x))/len(x)
#     y = y[::-1]
#     ax[1].errorbar(x/np.amax(x), y, xerr = abs(xe), c = c, fmt = 'o',markersize=12)
# labels = [temp + ' K' for temp in temps]
# ax[1].set_yticks([0,0.5,1])
# ax[1].legend(labels, bbox_to_anchor=(-0.1, 0.025, 1.1, 1), title = 'T', frameon=False, framealpha = 1, edgecolor = 'none')

# saveName = f'total_fig_{dateToSave()}'
# fig.savefig(f'{saveName}.svg', dpi=1200)

fig, ax = plt.subplots(figsize = (8,6))
ax.set_ylabel(r'P(x)')
ax.set_xlabel('x')
ax.set_ylim([0,0.8])
cats =  ['top', 'bottom', 'both', 'either', 'neither']
for kw,col in zip(['fm_heat', 'af_cool'], ['r','b']):
    recs.generateProbability(kw, cats)
    ax.errorbar(np.arange(len(cats)), recs.probs[1], yerr = recs.probs[2], c = col, fmt = 'o', markersize = 8)
cats2 =  ['', 'Top \nSurface', 'Bottom \nSurface', 'Both \nSurfaces', 'Either \nSurface', 'Neither \nSurface']
ax.set_xticklabels(cats2)
tags = ['FM Domains on Heating', 'AF Domains on Cooling']
fig.legend(tags,bbox_to_anchor=(0, 0, 0.7, 0.85), frameon=True, framealpha = 1, edgecolor = 'none')
saveName = f'P_Python_{dateToSave()}'
fig.savefig('{}.svg'.format(saveName), dpi=1200)
