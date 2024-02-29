# -*- coding: utf-8 -*-
"""
Predict Discrete values
@author: TSE
"""

# =============================================================================
# UserDefined Function for Plotting Classification output
# =============================================================================
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "green"))
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],\
        alpha=0.8, c=cmap(idx),\
        marker="+", label=cl)


# =============================================================================
# Import data 
# =============================================================================

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
Support Vector Machine
"""
# =============================================================================
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')

classifier.fit(X_train,y_train)

# =============================================================================
# Model Testing, confusion matrix, accuracy score
# =============================================================================
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)


# =============================================================================
# Add On Plot
# =============================================================================
plot_decision_regions(X_test, y_test, classifier)












