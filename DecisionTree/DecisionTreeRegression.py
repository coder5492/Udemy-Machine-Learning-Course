#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:29:20 2018

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
from sklearn.tree import DecisonTreeRegressor
regressor =  Dec

#visualising by plotting 
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X_poly))