#-*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:51:28 2019

@author: Suriya Prakash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def predict(d,h):
    datas=pd.read_csv('D://Downloads//%s.csv'%d)
    X = datas.iloc[:,0:1].values 
    y = datas.iloc[:,1].values
    from sklearn.linear_model import LinearRegression  
    from sklearn.preprocessing import PolynomialFeatures
    #from sklearn.metrics import r2_score
    poly = PolynomialFeatures(degree=8) 
    X_poly = poly.fit_transform(X) 
  
    poly.fit(X_poly, y) 
    lin = LinearRegression() 
    lin.fit(X_poly, y) 
    plt.scatter(X, y, color = 'blue') 
  
    plt.plot(X, lin.predict(poly.fit_transform(X)), color = 'red') 
    plt.title('Polynomial Regression') 
    plt.xlabel('hour') 
    plt.ylabel('count')
    plt.show()
    
    val=lin.predict(poly.fit_transform([[h]]))
    #r2=r2_score(y,val)
    #return(r2)
    return(val)
    
    
    
print("Day and Hour")
d=input()
h=float(input())
val1=predict(d,h)
if(val1 < 0):
    print("0")
else:
    print("demand: %d"%val1)
#print("%0.3f"%r)
    
