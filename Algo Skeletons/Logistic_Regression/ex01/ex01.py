# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ex01.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mndhlovu <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/12/12 16:00:48 by mndhlovu          #+#    #+#              #
#    Updated: 2018/12/12 17:26:35 by mndhlovu         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#IMPORT THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#import the dataset and the extract the dependant and independant variables
social_network = pd.read_csv('SocialNetworkAds.csv')
X = social_network.iloc[:, [2,3]].values
y = social_network.iloc[:, 4].values

#Visualize the data through the use of a heat map
sns.heatmap(social_network.corr())

#Split the data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling to improve
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fit the logistic regression to the training the dataset

from sklearn.linear_model import LogisticRegression
logR = LogisticRegression(randome_state = 0)
logR.fit(X_train, y_train)

#Predicting the test set results
y_pred = logR.predict(X_test)
#y_pred uncomment if you would want to view the test results

#visualization of the test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
        mp.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
            c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#model evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
