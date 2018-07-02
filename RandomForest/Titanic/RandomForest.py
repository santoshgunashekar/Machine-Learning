# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:14:25 2018

@author: Santosh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score



#data_test=pd.read_csv("test.csv")
#data_test['Sex']=le.fit_transform(data_test['Sex'].astype(str))
#data_test['Embarked']=le.fit_transform(data_test['Embarked'].astype(str))
#data_test.drop(['PassengerId'],axis=1,inplace=True)
#data_test.drop(['Name'],axis=1,inplace=True)
#data_test.drop(['Ticket'],axis=1,inplace=True)
#data_test.drop(['Age'],axis=1,inplace=True)
#data_test.drop(['Cabin'],axis=1,inplace=True)

data=pd.read_csv("train.csv")
le=LabelEncoder()
data['Sex']=le.fit_transform(data['Sex'].astype(str))
data['Embarked']=le.fit_transform(data['Embarked'].astype(str))
data.drop(['PassengerId'],axis=1,inplace=True)
data.drop(['Name'],axis=1,inplace=True)
data.drop(['Ticket'],axis=1,inplace=True)
data.drop(['Cabin'],axis=1,inplace=True)
data.drop(['Age'],axis=1,inplace=True)


#data_test['']
X=data.iloc[:,1:]
print(X.describe())
Y=data['Survived'].ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

model=RandomForestClassifier().fit(X_train, Y_train)
predict = model.predict(X_test)


print("Efficiency is :")
print(accuracy_score(predict, Y_test)*100)  

#count=0
#for i in range(len(Y_test)):
    #if(Y_test[i]==predict[i]):
        #count=count+1

      
#print((count/len(Y_test))*100)