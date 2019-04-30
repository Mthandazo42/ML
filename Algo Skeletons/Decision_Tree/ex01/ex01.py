# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ex01.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mndhlovu <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/12/12 17:32:19 by mndhlovu          #+#    #+#              #
#    Updated: 2019/04/29 15:38:23 by mndhlovu         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#import the data set and extract the independant and dependant variables
kyphosis = pd.read_csv('kyphosis.csv')
X = kyphosis.drop('Kyphosis', axis=1)
y = kyphosis['Kyphosis']

#Data analysis
sns.barplot(x='Kyphosis', y='Age',data=kyphosis)
sns.pairplot(kyphosis, hue='Kyphosis',palette='Set1')

#Dataset visualization
plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Kyphosis',data=kyphosis,palette='Set1')

#Splitting the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)

#Training the Dataset
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
        max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,
        random_state=None, slitter='best')
predictions = dtree.predict(X_test)
predictions

#Use the confusion matrix to evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))

#Random Forest Code usage
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
