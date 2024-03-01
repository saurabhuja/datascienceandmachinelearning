# -*- coding: utf-8 -*-
"""
Natural Lang Processing
@author: TSE
"""

# =============================================================================
# Import Libraries & Dataset
# =============================================================================
import numpy as np
import pandas as pd

dataset = pd.read_csv(r"moviereviews.tsv", delimiter='\t')


# Encoding label Column

from sklearn.preprocessing import LabelEncoder
labelobj = LabelEncoder()
dataset['label'] = labelobj.fit_transform(dataset["label"])

# Check for nan value
print("Number of nan in each column ", dataset.isna().sum())

# Drop the Rows containing nan
dataset.dropna(inplace=True)

#Identify blank rows
blanks = []

for i,lb,rv in dataset.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
            
# Drop the blank             
dataset.drop(blanks, inplace=True)

# =============================================================================
# Importing nltk libraries
# =============================================================================
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#stopset = set(stopwords.words('english'))
#stopset = set(stopwords.words('spanish'))
#stopset = set(stopwords.words('german'))

stopset = set(stopwords.words('english')) - set(('over','under','not','only','once','most'))


# =============================================================================
# Cleaning of Text
# =============================================================================
corpus = []

for i in range(0, len(dataset)):
    #Step1: Substitute all non-alphabets with a space , using python re package 
    review = re.sub('[^a-zA-Z]',' ', dataset.iloc[i,1])
    
#    Step2: Convert the review to lower case , python lower()
    review = review.lower()

#   Step3: Convert the review to tokens of words , using Split() method in python    
    review = review.split()
    
    ps = PorterStemmer()

#   Step4:Eliminate Stopwords , using nltk stopwords 
#   Step5: Stemming of words , using nltk stopwords     
    review = [ps.stem(word) for word in review if not word in stopset]

#   Step6: join these words to a stmt, using join method    
    review = ' '.join(review)
    
    corpus.append(review)
    
    
# =============================================================================
# Creating Number matrix from Text
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0].values

# =============================================================================
# Split dataset into training ans testing
# test_size = 0.20 , random_state = 0
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# =============================================================================
# Model implementation
# =============================================================================
from sklearn.svm import LinearSVC
classifier = LinearSVC()

classifier.fit(X_train,y_train)

# =============================================================================
# Model testing , pass X_test , confusion matrix, accuracy_score
# =============================================================================
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)


















