import numpy as np
import pandas as pd
from sklearn import naive_bayes
from urllib.request import urlopen
from sklearn.metrics import roc_auc_score
dataset=pd.read_csv("shika.csv", names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq','ar','as','at','au','av','aw','ax','ay','az','ba','bb','bc','bd','be','bf'])
print(dataset)

#print(dataset[0])
X=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1].ravel()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42)
y_train=y_train.ravel()
y_test=y_test.ravel()



clf=naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))