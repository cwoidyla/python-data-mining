# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 18:12:17 2016

@author: Conrad
"""

# %% Save data
import pickle
localVars = dir()
with open('labels.pickle', 'w') as f:
    pickle.dump(labels, f)
# %% Retrieve data
import pickle
with open('labels.pickle') as f:
    labels = pickle.load(f)