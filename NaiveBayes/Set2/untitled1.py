# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:32:56 2018

@author: Santosh
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopset=set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
df1=pd.read_csv("data.csv",names=['txt','category'])
df=df1.iloc[:350000,:]
df=df[::10]

vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
y=df.category
X=vectorizer.fit_transform(df.txt)
print(y.shape)
print(X.shape)
#X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
clf=naive_bayes.MultinomialNB()
clf.fit(X,y)
#predicted=clf.predict(X_test)
#count=0
#for i in range(len(predicted)):
#    if (predicted[i])==(y_test.iloc[i]):
#        count+=1
#print("Efficiency is %f",count/len(predicted)*100)
df2=df1.iloc[350000:,:]
df2=df2[::10]

vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
y2=df2.category
X2=vectorizer.fit_transform(df2.txt)
print(y2.shape)
print(X2.shape)
print(len(y2))
predicted=clf.predict(X2)
count=0
for i in range(len(X2)):
    if (predicted[i])==(y2.iloc[i]):
        count+=1
print("Efficiency is %f",count/len(X2)*100)