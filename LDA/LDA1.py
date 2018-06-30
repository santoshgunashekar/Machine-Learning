# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:33:40 2018

@author: Santosh
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plot_lda(X):

    for label in range(4):
        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label])
    plt.xlabel('Reduced Feature 1')
    plt.ylabel('Reduced Feature 2')
    plt.title('Wine classification using Linear Discriminant Analysis') 
    plt.grid()
    plt.tight_layout
    plt.show()
    

data = pd.read_csv('wine1.csv', delimiter=',')
y = data.iloc[:, 0]
X =data.iloc[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=0)

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train,Y_train)
X_reduced=lda.transform(X)
predict=lda.predict(X_test)
print("Predicted Classes\n",predict)
print('\n')
print("Actual Classes\n",Y_test.ravel())
print("Efficiency is ",accuracy_score(predict,Y_test)*100)

plot_lda(X_reduced)


