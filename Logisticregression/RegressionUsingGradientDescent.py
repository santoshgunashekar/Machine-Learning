# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:26:22 2018

@author: Santosh
"""
#upload necessary libraries that will be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl

# import the data into the program and divide them into input and output
data_address="train.csv"
data=pd.read_csv(data_address)
data=data.sort_values(by=['Y'])
xtrain=data.iloc[:,0:3]
ytrain=data.iloc[:,3].values


for i in range(1, len(ytrain)):
    if ytrain[i-1]==1:
        number_zeros=i
        break
    
    
for i in range(1,number_zeros):
    mpl.plot(xtrain.X1.iloc[i-1],xtrain.X2.iloc[i-1],'rx')  
for i in range(number_zeros,len(ytrain)-1):
    mpl.plot(xtrain.X1.iloc[i],xtrain.X2.iloc[i],'bo')    

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient_Descent(theta, alpha, x , y):
    m = x.shape[0]
    h = sigmoid(np.matmul(x, theta))
    grad = np.matmul(x.T, (h - y)) / m;
    theta = theta - alpha * grad
    return theta

theta = np.zeros(3)
iterations=1500
alpha = 0.1943

for i in range(iterations):
    theta = gradient_Descent(theta, alpha, xtrain, ytrain)
    print(theta)
  
print(theta)
