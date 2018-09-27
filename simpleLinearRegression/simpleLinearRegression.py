#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:15:06 2018

@author: sangeeth
"""

#Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3 ,random_state = 0)

#building linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test results
Y_pred = regressor.predict(X_test) 

#visualising the test resuts
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test))
plt.title('Salary Vs Expreience Training Set')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()