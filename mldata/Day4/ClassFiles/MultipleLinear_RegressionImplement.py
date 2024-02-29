# -*- coding: utf-8 -*-
"""
Predict Continuous Values-Reimplementaion only with 2 X features (RND & Mkt)
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
X = dataset.iloc[:,[0,2]].values       # X 2darray
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
mse = mean_squared_error(y_test,y_pred)    # old with 5 features 83502864.03257737
                                           # new with 2 features 67220275.37568113
                                           
rmse = np.sqrt(mse)                       # old with 5 features 9137.990152794944
                                          # new with 2 features 8198.797190788484
                                          
score = r2_score(y_test,y_pred)           # old with 5 features 0.9347068473282425
                                          # new with 2 features 0.9474386447268489



























