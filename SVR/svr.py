#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:00:59 2018

@author: sangeeth
"""

#Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler();
Y = sc_Y.fit_transform(Y)
X = sc_X.fit_transform(X)

#fitting the SVR model to the dataset 
from sklearn.svm import SVR
regressor =SVR(kernel = 'rbf')
regressor.fit(X,Y)

#predicting the outcome 
sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising by plotting 
plt.scatter(X,Y)
plt.plot(X,regressor.predict(X))