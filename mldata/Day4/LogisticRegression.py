# -*- coding: utf-8 -*-
"""
Predict Discrete values
Logistic Regression
@author: TSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("04SUV_Ad.csv")

# =============================================================================
# Feature Extraction
# =============================================================================
X = dataset.iloc[:,[2,3]].values     # 2d array matrix
y = dataset.iloc[:,4].values         # 1d array

# =============================================================================
# Train Test Split, test_size=0.25
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# =============================================================================
# Scaling of Values
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()

scObj.fit(X_train)         # Learn the scale / scaling on train data

X_train= scObj.transform(X_train)
X_test = scObj.transform(X_test)


# =============================================================================
# Model Implementation
"""
Logistic Regression
"""
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing, confusion matrix, accuracy score
# =============================================================================
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)





















