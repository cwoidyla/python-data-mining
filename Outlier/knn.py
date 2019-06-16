# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:08:37 2016

@author: Conrad
"""
# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mPatch
from matplotlib.legend_handler import HandlerLine2D
# %%
#knn function gets the dataset and calculates K-Nearest neighbors and distances
def knn(df,k):
    nbrs = NearestNeighbors(n_neighbors=3)
    nbrs.fit(df)
    distances, indices = nbrs.kneighbors(df)
    return distances, indices
# %%
#reachDist calculates the reach distance of each point to MinPts around it
def reachDist(df,MinPts,knnDist):
    nbrs = NearestNeighbors(n_neighbors=MinPts)
    nbrs.fit(df)
    distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts
# %%
#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))
# %%
#Finally lof calculates lot outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof
# %%
#We flag anything with outlier score greater than 1.2 as outlier#This is just for charting purposes
def returnFlag(x):
    if x['Score']>1.2:
       return 1
    else:
       return 0
# %%
#Read the file to data frame
data = pd.read_csv('codedDiamonds.csv')

#You can change below value for different MinPts
m=15
# %%
knndist, knnindices = knn(data,3)
# %%
reachdist, reachindices = reachDist(data,m,knndist)
# %%
irdMatrix = lrd(m,reachdist)
# %%
lofScores = lof(irdMatrix,m,reachindices)
# %% 
scores= pd.DataFrame(lofScores,columns=['Score'])
# %%
mergedData=pd.merge(data,scores,left_index=True,right_index=True)
mergedData['flag'] = mergedData.apply(returnFlag,axis=1)
Outliers = mergedData[(mergedData['flag']==1)]
Normals = mergedData[(mergedData['flag']==0)]

#Below section creates the charts

line1, = plt.plot([1], marker='o', label='Regular',linestyle='None',color='blue')
line2, = plt.plot([1], marker='*', label='Outlier',linestyle='None',color='red')

fig=plt.figure(dpi=80, facecolor='w', edgecolor='k')
fig.legend((line2,line1),('Outliers','Regular'),loc=1,numpoints=1,ncol=2)

#First we draw Carat vs Table vs Price#We show outliers with *

ax1 = plt.subplot2grid((1,1), (0,0),projection='3d')
ax1.scatter(Outliers['carat'],Outliers['table'],Outliers['price']/1000,c='r',marker='*')
ax1.scatter(Normals['carat'],Normals['table'],Normals['price']/1000,c='b',marker='o')
ax1.set_xlabel('Carat')
ax1.set_ylabel('Table')
ax1.set_zlabel('Price(K)')
ax1.set_title('Outliers Vs. Rest\nCarat, Table, Price View')

plt.tight_layout()
plt.show()

#Next we draw X vs Y vs Z#We show outliers with *
fig=plt.figure(dpi=80, facecolor='w', edgecolor='k')
fig.legend((line2,line1),('Outliers','Regular'),loc=1,numpoints=1,ncol=2)


ax2 = plt.subplot2grid((1,1), (0,0),projection='3d')
ax2.scatter(Outliers['x'],Outliers['y'],Outliers['z'],c='r',marker='*')
ax2.scatter(Normals['x'],Normals['y'],Normals['z'],c='b',marker='o')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Outliers Vs. Rest\nX,Y, Z View')

plt.tight_layout()
plt.show()

#Finally we draw the histogram of scores

fig=plt.figure(dpi=80, facecolor='w', edgecolor='k')


ax3 = plt.subplot2grid((1,1), (0,0))
ax3.hist(mergedData['Score'],bins=100,facecolor='cornflowerblue')
ax3.set_xlabel('LOF Score')
ax3.set_ylabel('Frequency')
ax3.set_title('Outlier Scores')



plt.tight_layout()
plt.show()