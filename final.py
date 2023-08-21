# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:17:35 2019

@author: HP
"""
import pickle
import pandas as pd
df = pd.read_csv("apy.csv")
df.drop(['Crop_Year'],axis=1,inplace=True)
df.drop(['Production'],axis=1,inplace=True)
from sklearn import preprocessing
print(df.columns)
le = preprocessing.LabelEncoder()
le1= preprocessing.LabelEncoder()
le2= preprocessing.LabelEncoder()
le3= preprocessing.LabelEncoder()
le.fit(df['State_Name'])

df.State_Name=le.transform(df["State_Name"])
le1.fit(df['Crop'])
#df.Crop=le1.transform(df["Crop"])

le2.fit(df['Season'])
#df.Season=le2.transform(df["Season"])

le3.fit(df['District_Name'])
#df.District_Name=le3.transform(df["District_Name"])
with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf_load = pickle.load(fid)
#print(clf_load.predict(train))
state=input("Enter state ")    
state=le.transform([state])
district=input("enter district ")
district=le3.transform([district])
area=int(input('Enter area in acres '))

season=input('Enter season: ')
#season=le2.transform([season])
season=[83]
data={'State_Name':state,'District_Name':district,'Season':season,'Area':area}
df=pd.DataFrame(data=data,index=[0])
ypred=clf_load.predict(df)
print(le1.inverse_transform(ypred))