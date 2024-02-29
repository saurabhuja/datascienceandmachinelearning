
# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("NewPC.csv")

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
# Grid Search For yper Parameter Tunning
# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Create dictionary with values to search the best value for parameters

parameters = {'n_estimators':[100,110,120,130,150],
              'min_samples_split':[4,6,8]}

gridObj = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                       param_grid=parameters,
                       scoring='neg_mean_squared_error')

# Fit data to find the best parameter

gridObj.fit(X_train,y_train)

# Display best Parameters

best_parameters = gridObj.best_params_

# Predictions using best Parameters

y_predGrid = gridObj.predict(X_test)

from sklearn.metrics import mean_squared_error

error_RFGrid = mean_squared_error(y_test,y_predGrid)








