#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[24]:


#Importing the datset
#Since this is a tab seperated file, I specifically had to give delimeter parameter. 
#Also, to ignore the texts in double quotes, I had to give quoting = '3'
dataset = pd.read_csv('Desktop/Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3)


# In[25]:


dataset


# In[26]:


#Cleaning the dataset for the 1st review and then for all reviews

import re
import nltk
nltk.download('stopwords') #downloaded a list of stop words from nltk library
from nltk.corpus import stopwords #imported the stopwords library
from nltk.stem.porter import PorterStemmer #Class for stemming
dataset['Review'][0] #1st review
corpus = [] #empty list that will hold clean review
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #Kept only alphabets and replaced all other characters with space
    review = review.lower() #Converted into lower case
    review = review.split() #Tokenization
    ps = PorterStemmer() #Created an object for Stemming class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    #Did stemming and removed stop words
    review = ' '.join(review) #rejoined the review from tokenized version
    corpus.append(review) #created a list of words from 1000 reviews


# In[27]:


corpus


# In[28]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#limiting the number of words to 1500 to reduce sparsity
X = cv.fit_transform(corpus).toarray() #Create a sparse matrix. This is the collection of all independent variables
y = dataset.iloc[:, 1].values #dependent variable


# In[31]:


#Using Naive Bayes to predict results

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[32]:


#Confusion Matrix for test set
cm


# In[41]:


#Using Random Forest to predict results

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[42]:


#Confusion Matrix for test set
cm


# In[ ]:




