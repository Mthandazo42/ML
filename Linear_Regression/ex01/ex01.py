# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ex01.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mndhlovu <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/12/12 14:58:42 by mndhlovu          #+#    #+#              #
#    Updated: 2018/12/12 15:33:05 by mndhlovu         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#IMPORT THE NECESSARY LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd;
import seaborn as sns

#IMPORT THE DATASET USING THE READ_CSV FUNCTIONS AND EXTRACT INDEPENDANT AND
#DEPENDANT VARIABLES FROM IT

salary_data = pd.read_csv('Salary_Data.csv')
X = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, 1].values

#VISUALIZE THE DATA
sns.distplot(salary_data['YearsExperience'],kde=False,bins=0)
sns.countplot(y='YearsExperience',data=salary_data)
sns.barplot(x='YearsExperience',y='Salary',data=salary_data)
sns.heatmap(salary_data.corr())

#SPLIT THE DATASET INTO TRAINING AND TEST DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#FITTING THE LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_pred = lr.predict(X_test)
y_pred

#VISUALIZE THE TRAIN RESULTS

plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.title('Salary ~ Experience (Train Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#VISUALIZE THE TEST RESULTS

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, lr.predict(X_train), color = 'red')
plt.title('Salary Vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#CALCULATING THE RESIDUALS
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,y_pred))
print('MSE:', metrics.mean_square_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))
