# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:25:50 2019

@author: HP
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
clf = SVC(gamma='auto')


le = preprocessing.LabelEncoder()
le1= preprocessing.LabelEncoder()
le2= preprocessing.LabelEncoder()
le3= preprocessing.LabelEncoder()

df = pd.read_csv("apy.csv")
df.drop(['Crop_Year'],axis=1,inplace=True)
df.drop(['Production'],axis=1,inplace=True)
print(df.columns)
le.fit(df['State_Name'])

df.State_Name=le.transform(df["State_Name"])
le1.fit(df['Crop'])
df.Crop=le1.transform(df["Crop"])

le2.fit(df['Season'])
df.Season=le2.transform(df["Season"])

le3.fit(df['District_Name'])
df.District_Name=le3.transform(df["District_Name"])

#df.drop(['Crop'])

train, test = train_test_split(df, test_size=0.5)
y_train = train['Crop']
y_test = test['Crop']
train.drop(['Crop'],axis=1,inplace=True)
test.drop(['Crop'],axis=1,inplace=True)
print('yes')
clf.fit(train, y_train)
with open('my_dumped_classifier.pkl', 'wb') as fid:
    pickle.dump(clf, fid)
with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf_load = pickle.load(fid)
#print(clf_load.predict(train))
ypred=clf_load.predict(test)
print(len(ypred))
print ("Number of mislabeled points : %d" % (y_test != ypred).sum())


#print(train['Crop'])
#print(df['Season'])
