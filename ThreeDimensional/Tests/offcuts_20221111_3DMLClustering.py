# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:13:57 2022

@author: massey_j
"""

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

x = finals['fm_heat'][finals['fm_heat']['temp'] == 310]
#x = x[['centroid-0', 'centroid-1','slice']].values

p = []
for s in np.unique(x['slice']):
    temp = x[x['slice'] == s]
    p.append(len(temp))
cluster_no = np.int(np.mean(np.array(p)))
#cluster_no = len(recs.finalFM[recs.finalFM['temp'] == 310])

x = x[['centroid-0', 'centroid-1','slice']].values
model = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(x)

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
for i in range(cluster_no): 
    ax.scatter(x[y_clusters == i,0],x[y_clusters == i,1],x[y_clusters == i,2], label = i)
    
"""size clustering"""
n = finals['fm_heat']
n = n[n['temp'] == 310]
n['x-size'] = abs(n['bbox-3']-n['bbox-0'])
n['y-size'] = abs(n['bbox-4']-n['bbox-1'])
n['z-size'] = abs(n['bbox-5']-n['bbox-2'])

x = n[['x-size', 'y-size','z-size']].values 

WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (7,7))
plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
plt.xticks(np.arange(11))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

cluster_no = 2
model = KMeans(n_clusters = cluster_no, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(x)

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
for i in range(cluster_no): 
    ax.scatter(x[y_clusters == i,0],x[y_clusters == i,1],x[y_clusters == i,2], label = i)
fig.legend()


def calcCentroid(x): 
    return np.min(x) + 0.5*(np.max(x)-np.min(x))

centroids = []
widths = []
for i in range(cluster_no): 
    centroids.append([calcCentroid(x[y_clusters == i,0]),calcCentroid(x[y_clusters == i,1]),calcCentroid(x[y_clusters == i,2])])
    widths.append(([calcWidth(x[y_clusters == i,0]),calcWidth(x[y_clusters == i,1]),calcWidth(x[y_clusters == i,2])]))
centroids = np.array(centroids)
widths = np.array(widths)

def calcCentroid(x): 
    return np.min(x) + 0.5*(np.max(x)-np.min(x))

def calcWidth(x): 
    return (np.max(x)-np.min(x))

#Touching top and bottom
for i in range(cluster_no): 
    if max(x[:,2]) is in x[y_clusters == i,2]: 
        print(i)
#max
t1 = centroids + widths/2
top = t1[:,2] == max(x[:,2])
t2 = centroids - widths/2
bottom = t2[:,2] < 0 