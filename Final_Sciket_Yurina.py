#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:03:37 2018

@author: YURI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as ptl
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Create Classifiers
LR = LogisticRegression()
LS = LinearSVC(C=1.0)
RFClass = RandomForestClassifier(n_estimators=100)

#Read data 
data = pd.read_csv("/Users/YURI/anaconda2/lib/python2.7/diabetes.csv")
data.head()

data.iloc[:,0:-1] 
#X = data[:,0:8]
#y = data[:, 8]
X = data[['Insulin', 'BMI']].values
y = data[['Outcome']].values
#Correlations test
corr = data.corr()
corr ['Outcome'].sort_values(ascending=False)

# Visualization (set1): Glucose vs Age
# They are related linearly
def visualize1(x1):
    fig, ax = ptl.subplots()
    ax.scatter(x1.iloc[:,5].values, x1.iloc[:,4].values,color='R')
    ax.set_title('BMI vs Insulin')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Insulin')

# disgard anomalies (outlier values)    
data[['BMI','Insulin']] = data[['BMI','Insulin']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize1(data)

#Visulization (set2): Insulin vs Glucose
# They are related linearly
def visualize2(x2):
    fig, ax = ptl.subplots()
    ax.scatter(x2.iloc[:,1].values, x2.iloc[:,4].values,color='Orange')
    ax.set_title('Glucose vs Insulin')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Insulin')

# disgard anomalies (outlier values)    
data[['Glucose','Insulin']] = data[['Glucose','Insulin']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize2(data)

# Visulization (set3) : Glucose vs BMI
# They are not related
def visualize3(x3):
    fig, ax = ptl.subplots()
    ax.scatter(x3.iloc[:,1].values, x3.iloc[:,5].values,color='gold')
    ax.set_title('Glucose vs BMI')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('BMI')

# disgard anomalies (outlier values)    
data[['Glucose','BMI']] = data[['Glucose','BMI']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize3(data)

# Visulization (set4) : Bloodpressure vs Insulin
# They are not related
def visualize4(x4):
    fig, ax = ptl.subplots()
    ax.scatter(x4.iloc[:,2].values, x4.iloc[:,4].values,color='springgreen')
    ax.set_title('BloodPressure vs Insulin')
    ax.set_xlabel('BloodPressure')
    ax.set_ylabel('Insulin')

# disgard anomalies (outlier values)    
data[['BloodPressure','Insulin']] = data[['BloodPressure','Insulin']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize4(data)

# Visulization (set5) : Bloodpressure vs BMI
# They are not related
def visualize5(x5):
    fig, ax = ptl.subplots()
    ax.scatter(x5.iloc[:,2].values, x5.iloc[:,5].values,color='dodgerblue')
    ax.set_title('BloodPressure vs BMI')
    ax.set_xlabel('BloodPressure')
    ax.set_ylabel('BMI')

# disgard anomalies (outlier values)    
data[['BloodPressure','BMI']] = data[['BloodPressure','BMI']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize5(data)

# Visulization (set6) : Bloodpressure vs Glucose
# They are not related
def visualize6(x6):
    fig, ax = ptl.subplots()
    ax.scatter(x6.iloc[:,2].values, x6.iloc[:,1].values,color='blueviolet')
    ax.set_title('BloodPressure vs Glucose')
    ax.set_xlabel('BloodPressure')
    ax.set_ylabel('Glucose')

# disgard anomalies (outlier values)    
data[['BloodPressure','Glucose']] = data[['BloodPressure','Glucose']].replace(0,np.NaN)
data.dropna(inplace = True)
visualize6(data)

# Visulization (set7): Pedigree function vs class
# They are not related
def visualize7(x7):
    fig, ax = ptl.subplots()
    ax.scatter(x7.iloc[:,6].values, x7.iloc[:,8].values,color='deeppink')
    ax.set_title('Pedigreefunction vs Outcome')
    ax.set_xlabel('DiabetesPedigreeFunction')
    ax.set_ylabel('Outcome')

# disgard anomalies (outlier values)    
#data[['DiabetesPedigreeFunction','Class']] = data[['DiabetesPedigreeFunction','Class']].replace(0,np.NaN)
#data.dropna(inplace = True)
visualize7(data)
# Feature scaling isolation 
sc = StandardScaler()
X = sc.fit_transform(X)

# 2-Dimentional numpy array, distribution.
mean = np.mean(X, axis=0)
print('Mean: (%d, %d)' % (mean[0], mean[1]))
standard_deviation = np.std(X, axis=0)
print('Standard deviation: (%d, %d)' % (standard_deviation[0], standard_deviation[1]))
print(X[0:20, :])
 
# split data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Make a prediction
diabetes = LogisticRegression()
diabetes.fit(X_train, y_train.ravel())
y_pred = diabetes.predict(X_test)

#Evaluate the performance
performance = confusion_matrix(y_test, y_pred)
print(performance)

def precision_recall(y_test, y_pred):
    performance = confusion_matrix(y_test, y_pred)
    
    
