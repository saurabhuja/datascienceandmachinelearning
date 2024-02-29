# -*- coding: utf-8 -*-
"""
DecisionTree Regressor - Regression Tree - Predicting Continuos Values
@author: TSE
"""

# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"NewPC.csv")

# =============================================================================
#  Feature Extraction
# =============================================================================
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values

# =============================================================================
# Train Test Split 80-20
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Model Implementation
# Fitting DT Regressor
"""
min_samples_split= 2    Error 39.9    Not stable, overfitting
min_samples_split= 4    Error 46.17    
min_samples_split= 5    Error 50.61
min_samples_split= 8    Error 96.50
"""
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(min_samples_split=5)

regressor.fit(X_train,y_train)

# =============================================================================
# Model Testing 
# =============================================================================
y_pred_TestSample1 = regressor.predict(X_test[0:5,:])
y_pred_TestSample2 = regressor.predict(X_test[5:10,:])

from sklearn.metrics import mean_squared_error
error_DTSample1 = mean_squared_error(y_test[0:5],y_pred_TestSample1)
error_DTSample2 = mean_squared_error(y_test[5:10],y_pred_TestSample2)

y_pred_train = regressor.predict(X_train[0:5,:])
error_DTrain = mean_squared_error(y_train[0:5],y_pred_train)

# =============================================================================
# Tree plotting
# =============================================================================

from sklearn.tree import export_graphviz

export_graphviz(regressor, out_file="treePC.dot", feature_names=["Average Salary"])













