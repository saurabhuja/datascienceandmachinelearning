# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:30:10 2024
Simple Linear Regression
@author: tsecl
"""

# =============================================================================
# Import Data & Libraries
# =============================================================================
import numpy as np
import pandas as pd

dataset = pd.read_csv("01HR_Data.csv")

# =============================================================================
# Extract X & Y Features
# =============================================================================
X = dataset.iloc[:,[0]].values    # 2d numpy array
y = dataset.iloc[:,1].values      # 1d numpy array

# =============================================================================
# Train-Test Split
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state = 0)

# =============================================================================
# Model Implementaion
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)   # Machine is learning b0 & b1

print("Slope b1",regressor.coef_)
print("Intercept b0",regressor.intercept_)

# =============================================================================
# Model Testing
# r2_score tells us the percentage of variation in y that can be explained by X features
# =============================================================================
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
score = r2_score(y_test,y_pred)  # 97.49%













