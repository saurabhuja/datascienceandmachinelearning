# -*- coding: utf-8 -*-
"""
Predict Continuous values
@author: TSE
"""
# =============================================================================
# Import Libraries & Dataset
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Pressure.csv')

# =============================================================================
# Extract Features
# =============================================================================
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values

# =============================================================================
# Train-Test Split
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3, random_state=0)

# =============================================================================
# Add Polynomial Degree to X features
# =============================================================================
from sklearn.preprocessing import PolynomialFeatures
polyObj = PolynomialFeatures(degree= 7, include_bias=False)

Xtrain_poly = polyObj.fit_transform(X_train)
Xtest_poly = polyObj.fit_transform(X_test)
# =============================================================================
# Model Implementation
#  fit_intercept = True   # Algo calculates intercept, default
#  fit_intercept = False   # Algo sets intercept to 0 or 1 as constant and doesnt calculate it

# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain_poly,y_train)

# =============================================================================
# Model Testing
# =============================================================================
y_pred = regressor.predict(Xtest_poly)

from sklearn.metrics import mean_squared_error
# error_2 = mean_squared_error(y_test,y_pred)
# error_5 = mean_squared_error(y_test,y_pred)
error_7 = mean_squared_error(y_test,y_pred)
# error_10 = mean_squared_error(y_test,y_pred)
# error_15 = mean_squared_error(y_test,y_pred)
# error_20 = mean_squared_error(y_test,y_pred)

# =============================================================================
# Single Prediction
# =============================================================================
userInput = 90
userInputDegree = polyObj.fit_transform([[90]])
prediction = regressor.predict(userInputDegree)
print("Predicted Value", prediction[0])

prediction = regressor.predict(polyObj.fit_transform([[60]]))
print("Predicted Value", prediction[0])


# =============================================================================
# Add-on Visualization
# =============================================================================
X_grid = np.arange(min(X_test), max(X_test),0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X_test, y_test, c="Red")
plt.plot(X_grid, regressor.predict(polyObj.fit_transform(X_grid)),c="Blue")


























