# -*- coding: utf-8 -*-
"""
Predict Continuous Values- ReImplement- 1 feature
@author: tsecl
"""

# =============================================================================
# Import data and Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("02Companies.csv")


# =============================================================================
# Extract X & Y Features
# =============================================================================
X = dataset.iloc[:,[0]].values       # X 2d numpy array
y = dataset.iloc[:,4].values     # 1d numpy array

# =============================================================================
# Train-Test Split
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)   # Machine is learning b0 & b1

print("Slobe b1", regressor.coef_)
print("Intercept b0",regressor.intercept_)

# =============================================================================
# Model Testing
# r2_score tells us the percentage of variation in y that can be explained by X features
# =============================================================================

y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred)   # 68473440.71905927
rmse = np.sqrt(mse)                       # 8274.868018225987
score = r2_score(y_test,y_pred)          # 0.946458760778722

















