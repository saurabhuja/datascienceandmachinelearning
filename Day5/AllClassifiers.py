# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:40:41 2024
Try out All Algorithms in Classification for Single Usa Case
@author: tsecl
"""

# =============================================================================
# Import Libraries & Dataset
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn Modelling-Bank Customers.csv")
# =============================================================================
# Extract X & Y Features 
# =============================================================================
X = dataset.iloc[:,3:13]       # X dataframe
y = dataset.iloc[:,13].values     # 1d numpy array

# =============================================================================
# Encoding of Geography & Gender Column 
# =============================================================================
X = pd.get_dummies(data=X, columns=['Geography','Gender'], drop_first=True)

X = X.values    # convert X dataframe to numpy array

# =============================================================================
# Train Test Split 80-20 split
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Scaling 
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()

scObj.fit(X_train)         # Learn the scale / scaling on train data

X_train= scObj.transform(X_train)
X_test = scObj.transform(X_test)

# =============================================================================
# Model Implementation
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

modelName = {0:'LR',1:'SVC',2:'NB',3:'DT',4:'RF',5:'KNN'}

scoreList = []

modelObjList = [LogisticRegression(),
                SVC(kernel='rbf'),
                GaussianNB(),
                DecisionTreeClassifier(max_depth=2),
                RandomForestClassifier(n_estimators=10,random_state=0),
                KNeighborsClassifier(n_neighbors=5)]



for i in range(len(modelObjList)):
    scoreList.append(accuracy_score(y_test, modelObjList[i].fit(X_train,y_train).predict(X_test)))

for i in range(len(scoreList)):
    print("Model {0} Accuracy {1} ".format(modelName[i],scoreList[i]))


# =============================================================================
# K-Fold Cross Validation
# =============================================================================

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=SVC(kernel='rbf'), X = X_train,y=y_train,cv=10)

print("Mean", accuracies.mean())
print("Std", accuracies.std())









