#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:13:15 2018

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

#fitting DecisionTree to dataset
from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators = 1000,random_state= 0)
regressor.fit(X,Y)

#predicting the salary
regressor.predict(6.5)

#visualising by plotting 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y)
plt.plot(X_grid,regressor.predict(X_grid))