# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:44:26 2023

@author: massey_j
"""
SMALL_SIZE = 16
MEDIUM_SIZE = 4
BIGGER_SIZE = 8

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=28)     # fontsize of the axes title
plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'mathtext.default':  'regular' })
plt.rcParams["legend.frameon"] = False

date = dateToSave()
#vol cool
# af_cool['logVol'] = np.log(af_cool['volnew'])
# ticks_cool = np.unique(np.round(af_cool['logVol'], 0))
# fin_cool = int(min(ticks_cool))
# af_cool['T'] = af_cool.apply(lambda row: str(int(row['temp']))+' K', axis =1)
# #temps = ['300', '330', '440']
# #labels = [temp + ' K' for temp in temps]
# ax = sns.histplot(data = af_cool, x = 'logVol', hue = 'T', palette='viridis')
# ax.set(xlabel='Log(Volume)', ylabel='Frequency', label = labels)
# ax.set_xticks(range(fin_cool, 1, 2), labels=np.arange(fin_cool, 1,2))
# #ax.legend(bbox_to_anchor=(0, 0.5, 1, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
# saveName = f'AF_VolumeAnalysis_{date}'
# fig = ax.get_figure()
# fig.savefig('{}_small.svg'.format(saveName), dpi=1200)

# # #domain cool
ax = sns.scatterplot(data = af_cool, x = 'xnew', y = 'znew', hue = 'T', palette='viridis', legend = False, s= 100)
ax.set(xlabel='x size (nm)', ylabel='z size (nm)')
#ax.legend(bbox_to_anchor=(0, 0.3, 1, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
saveName = f'AF_domainAnalysis_XZ_{date}'
fig = ax.get_figure()
fig.savefig('{}_small.svg'.format(saveName), dpi=1200)

# ax = sns.scatterplot(data = af_cool, x = 'xnew', y = 'ynew', hue = 'T', palette='viridis', legend = False, s= 100)
# ax.set(xlabel='x size (nm)', ylabel='y size (nm)')
# #ax.legend(bbox_to_anchor=(0, 0.3, 1, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
# saveName = f'AF_domainAnalysis_XY_{date}'
# fig = ax.get_figure()
# fig.savefig('{}_small.svg'.format(saveName), dpi=1200)

# #vol heat
# fm_heat['logVol'] = np.log(fm_heat['volnew'])
# fm_heat['T'] = fm_heat.apply(lambda row: str(int(row['temp']))+' K', axis =1)
# ticks = np.unique(np.round(fm_heat['logVol'], 0))
# fin = int(min(ticks))
# # temps = ['310', '335', '375']
# # labels = [temp + ' K' for temp in temps]
# ax = sns.histplot(data = fm_heat, x = 'logVol', hue = 'T', palette='CMRmap')
# ax.set(xlabel='Log(Volume)', ylabel='Frequency')
# ax.set_xticks(range(fin, 1, 2), labels=np.arange(fin, 1,2))
# #ax.legend(labels, bbox_to_anchor=(0, 0.5, 0.7, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
# saveName = f'FM_volumeAnalysis_{date}'
# fig = ax.get_figure()
# fig.savefig('{}_small.svg'.format(saveName), dpi=1200)

# # #domain heat
# ax = sns.scatterplot(data = fm_heat, x = 'xnew', y = 'znew', hue = 'T', palette='CMRmap', legend = False, s= 100)
# ax.set(xlabel='x size (nm)', ylabel='z size (nm)')
# #ax.legend(bbox_to_anchor=(0, 0.3, 1, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
# saveName = f'FM_domainAnalysis_XZ_{date}'
# fig = ax.get_figure()
# fig.savefig('{}_small.svg'.format(saveName), dpi=1200)

# ax = sns.scatterplot(data = fm_heat, x = 'xnew', y = 'ynew', hue = 'T', palette='CMRmap', legend = False, s = 100)
# ax.set(xlabel='x size (nm)', ylabel='y size (nm)')
# #ax.legend(bbox_to_anchor=(0, 0.3, 1, 0.5), title = 'T', frameon=True, framealpha = 1, edgecolor = 'none')
# saveName = f'FM_domainAnalysis_XY_{date}'
# fig = ax.get_figure()
# fig.savefig('{}_small.svg'.format(saveName), dpi=1200)
