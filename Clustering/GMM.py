# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 21:55:57 2016

@author: Conrad
"""

# %% import relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from sklearn.metrics.cluster import homogeneity_score, normalized_mutual_info_score
import numpy as np
import os, csv, time
from operator import itemgetter # helps with sorting
# %% change to data directory
print(os.getcwd())
os.chdir('C:\Users\Owner\Desktop\DM_Proj\Clustering')
print(os.getcwd())
# %% All mesonet data ordered
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
# %%

# %%
#labels.remove('STID')
labels = np.array(labels)
labels = np.delete(labels, 0) # remove first element in array
#labels = labels.astype(np.float)
# %% View data thus far
for i in range(3):
    print "data " + str(data[i+1]) + "\n"
    print "labels " + str(labels[i]) + "\n"
    print "trim" + str(trimData[i]) + "\n"
# %%
start = time.time()
gmix = mixture.GaussianMixture(n_components=6, covariance_type='full')
gmix.fit(trimData)
end = time.time()
print(end - start)

print gmix.means_
predictions = gmix.predict(trimData)
# %%view data
kylabels = pd.DataFrame({'a':labels})
kylabels.head(10)
kylabels.tail(10)
kylabels['a'].value_counts()
# %%

# %%
cllabels = pd.DataFrame({'a':predictions})
cllabels.head(10)
cllabels.tail(10)
cllabels['a'].value_counts()
# %%
indexRange = np.where(labels=="1")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
indexRange = np.where(labels=="2")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
indexRange = np.where(labels=="3")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
indexRange = np.where(labels=="4")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
indexRange = np.where(labels=="5")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
indexRange = np.where(labels=="6")
cllabels = pd.DataFrame({'a':predictions[indexRange]})
cllabels['a'].value_counts()
sum(cllabels['a'].value_counts())
# %% Metrics
# normalized_mutual_info_score(labels_true, labels_pred)
normalized_mutual_info_score(labels, predictions) # overall
# %%
indexRange = np.where(labels=="1")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # FARM
indexRange = np.where(labels=="2")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # GRHM
indexRange = np.where(labels=="3")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # HTFD
indexRange = np.where(labels=="4")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # MRRY
indexRange = np.where(labels=="5")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # PGHL
indexRange = np.where(labels=="6")
normalized_mutual_info_score(labels[indexRange], kmeans.labels_[indexRange]) # RSVL