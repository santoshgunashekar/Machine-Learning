# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:18:37 2018

@author: Santosh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# import some data to play with
iris = np.loadtxt('ex2data1.txt', delimiter=',')
X = iris[:, 0:2]  # we only take the first two features.
y = iris[:,2]

lr = linear_model.LogisticRegression(C=10)
lr.fit(X, y)
print(lr.predict(X[0:len(y),:]))