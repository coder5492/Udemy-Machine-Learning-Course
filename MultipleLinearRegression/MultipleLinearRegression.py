#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:24:13 2018

@author: sangeeth
"""

#Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#taking care of categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:,1:]

#splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

# =============================================================================
# #fitting the model for MultipleLinearRegression
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train,Y_train)
# 
# #predicting the results
# Y_pred = regressor.predict(X_test)
# =============================================================================

#Backward Elimination
import statsmodels.formula.api as sm
#appending a column of 1's at the beginning
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 )
Xopt = X[:,[0,3]]
est = sm.OLS(Y, Xopt)
est2 = est.fit()
print(est2.summary())
