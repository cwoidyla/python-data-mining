# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:36:00 2016

@author: Conrad
"""
# %% load relevant libraries
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import os, csv
# %% change to data directory
print(os.getcwd())
os.chdir('C:\Users\Owner\Desktop\DM_Proj\Clustering')
print(os.getcwd())
# %% load csv data
data = []
with open('allMeso.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
# %% create variables for data
trimData = []
labels = []
for row in data:
    trimData.append(row[1:-1])
    labels.append(row[-1])
trimData = np.array(trimData[1:],dtype=np.float)
# %% Perform hierarchical clustering algorithm
start = time.time()
fit = linkage(trimData, 'ward')
end = time.time()
print(end - start)
# %% Dendrogram
plt.figure(figsize=(25,10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(
  fit,
  truncate_mode='lastp',
  p=20,
  show_leaf_counts=False,
  leaf_rotation=90., # rotates the x axis label)
  leaf_font_size=12., # font size for the x axis labels
  show_contracted=True,
)
plt.show()
# %% view last 10 merges
fit[-10:,2]



