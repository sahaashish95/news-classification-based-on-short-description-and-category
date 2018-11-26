# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:43:54 2018

@author: ashish saha
"""
# Importing the libraries
import json as j
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re

#splitting data in to categories and convert in to dataframe
json_data=None
with open('News_Category_Dataset.json')as data_file:
    lines=data_file.readlines()
    joined_lines="[" + ",".join(lines)+ "]"
    json_data=j.loads(joined_lines)
dataset=pd.DataFrame(json_data)
dataset['headline+ description']=dataset['headline'] + ' ' + dataset['short_description']#adding headline and short description coloumn



X=dataset.iloc[:,6].values#choosing 6th coloumn
X=pd.DataFrame(X) #convert in to dataframe
X=X.rename(columns={0:'text'})#renaming the coloumn


#choosing category coloumn from dataset
y=dataset.iloc[:,1].values
y=pd.DataFrame(y)#convert y into dataframe


#importing all libraries for cleaning text
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

stemmer = SnowballStemmer('english')
words = stopwords.words("english")


#cleaning the dataset
X['cleaned'] = X['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())    

#spliting data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X['cleaned'], y, test_size=0.2)

'''making model for traiing and testing, in this we create bag of words after that it goes linear svc algorithm'''
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])
    
#fit pipeline model in to traing and test data
model = pipeline.fit(X_train, y_train)

#fit test data for getiing prediction based on test data
y_pred=model.predict(X_test)
y_pred=y=pd.DataFrame(y_pred)#convert in to dataframe

print("accuracy score: " + str(model.score(X_test, y_test)))#getting accuracy


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    



