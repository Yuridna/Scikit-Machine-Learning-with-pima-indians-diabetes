#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:03:37 2018

@author: YURI
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
# Create Classifiers
LR = LogisticRegression()
LS = LinearSVC(C=1.0)
RFClass = RandomForestClassifier(n_estimators=100)

# Read data 
data = pd.read_csv("/Users/YURI/anaconda2/lib/python2.7/diabetes.csv")

X = data[['Glucose', 'BMI']].values
y = data[['Outcome']].values

# Correlations test
corr = data.corr()
corr ['Outcome'].sort_values(ascending=False)

# Pearson correlation
correlations = data.corr(method='pearson')
print(correlations)

# Data Plots 
sns.set(style='white')
plt.figure(figsize=(20,20))
sns.pairplot(data, hue='Outcome', palette= "RdPu")

foo = []
#Make the list for each row)
foo.append('Pregnancies')
foo.append('Glucose')
foo.append('BloodPressure')
foo.append('SkinThickness')
foo.append('Insulin')
foo.append('BMI')
foo.append('DiabetesPedigreeFunction')
foo.append('Age')
foo.append('Outcome')

ax = []
for j in range(7):  # do the following 7 times:
    ax.append('ax'+str(j))

col = []
#Create the color list (rainbow)
col.append("R")
col.append("Orange")
col.append("gold")
col.append("springgreen")
col.append("dodgerblue")
col.append("blueviolet")
col.append("deeppink")

#y = data
#j = index of figures (j is integer 0<j<7)
#xl = xlabel (string)
#yl = ylabel (string)
#define visulize function to obtain correlation between two quantities.

xval = 0
yval = 0
def visualize(y, j, xl, yl):
    
    fig, ax[j] = plt.subplots()

    global xval
    global yval
    
    for k in range(9):
        if foo[k] == xl:
            xval = y.iloc[:,k].values
        if foo[k] == yl:
            yval = y.iloc[:,k].values
        
    ax[j].scatter(xval,yval,color=col[j])
    ax[j].set_title('%s vs %s' %(xl,yl))
    ax[j].set_xlabel(xl)
    ax[j].set_ylabel(yl)
    data[[xl,yl]] = data[[xl,yl]].replace(0,np.NaN)
    data.dropna(inplace = True)
        
# Visualization (set1): BMI vs Insulin
# They are related linearly
visualize(data, 0, 'BMI', 'Insulin')

# Visulization (set2): Glucose vs Insulin
# They are related linearly
visualize(data, 1, 'Glucose', 'Insulin')

# Visulization (set3) : Glucose vs BMI
# They are not related
visualize(data, 2, 'Glucose', 'BMI')

# Visulization (set4) : BloodPressure vs Insulin
# They are not related
visualize(data, 3, 'BloodPressure', 'Insulin')

# Visulization (set5) : BloodPressure vs BMI
# They are not related
visualize(data, 4, 'BloodPressure', 'BMI')

# Visulization (set6) : BloodPressure vs Glucose
# They are not related
visualize(data, 5, 'BloodPressure', 'Glucose')

# Visulization (set7): Pedigree function vs class
# They are not related
visualize(data, 6, 'DiabetesPedigreeFunction', 'Outcome')


# Plasma Glucose Level Test
print(data[data.Glucose == 0].shape[0])
print(data[data.Glucose == 0].index.tolist())
print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())

# BMI Range Test
print(data[data.BMI == 0].shape[0])
print(data[data.BMI == 0].index.tolist())
print(data[data.BMI == 0].groupby('Outcome')['Age'].count())

# Read data #2 
data = pd.read_csv("/Users/YURI/anaconda2/lib/python2.7/diabetes.csv")

# Visualize(Specific-Glucose & BMI Level)
plt.figure(figsize=(20,5))
glucose_algorithm = data.groupby('Glucose').Outcome.mean().reset_index()
sns.barplot(glucose_algorithm.Glucose, glucose_algorithm.Outcome)
plt.title('% of Diagnosed by Glucose Reading')
plt.show()

plt.figure(figsize=(14,3))
glucose_algorithm = data.groupby('Glucose').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['Glucose'], color='navy', kde=False, label='0 Class')
sns.distplot(data[data.Outcome == 1]['Glucose'], color='Red', kde=False, label='1 class')
plt.legend()
plt.title('PIMA Indians Glucose Values')
plt.show()

plt.figure(figsize=(20,5))
BMI_algorithm = data.groupby('BMI').Outcome.mean().reset_index()
sns.barplot(BMI_algorithm.BMI, BMI_algorithm.Outcome)
plt.title('% of Diagnosed by BMI Reading')
plt.show()

plt.figure(figsize=(14,3))
BMI_algorithm = data.groupby('BMI').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['BMI'], color='navy', kde=False, label='Class 0')
sns.distplot(data[data.Outcome == 1]['BMI'], color='Red', kde=False, label='Class 1')
plt.legend()
plt.title('PIMA Indians BMI Values')
plt.show()

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

# Evaluate the performance
performance = confusion_matrix(y_test, y_pred)
print(performance)


    
    
