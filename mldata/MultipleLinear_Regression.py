# -*- coding: utf-8 -*-
"""
Predict Continuous Values
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
X = dataset.iloc[:,0:-1]       # X dataframe
y = dataset.iloc[:,4].values     # 1d numpy array

# =============================================================================
# Encoding of State Column
# =============================================================================
X = pd.get_dummies(data=X, columns=['State'], drop_first=True)

X = X.values    # convert X dataframe to numpy array
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
mse = mean_squared_error(y_test,y_pred)   
rmse = np.sqrt(mse)              
score = r2_score(y_test,y_pred)   

# =============================================================================
# Feature Evaluation
# ['RNDSpend', 'Administration', 'MarketingSpend', 'Profit',
       # 'State_Florida', 'State_New York']
# =============================================================================

datasetNew = pd.read_csv("02Companies.csv")
datasetNew = pd.get_dummies(data=datasetNew, columns=['State'], drop_first=True)

                                                      # Pearsonr
datasetNew['RNDSpend'].corr(datasetNew['Profit'])     # 0.9729004656594832
datasetNew['Administration'].corr(datasetNew['Profit'])  #
datasetNew['MarketingSpend'].corr(datasetNew['Profit'])  #
datasetNew['State_Florida'].corr(datasetNew['Profit'])   #
datasetNew['State_New York'].corr(datasetNew['Profit'])  #
























