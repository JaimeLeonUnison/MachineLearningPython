#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
print('Dataset imported successfully')
print(X)

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('Categorical data encoded successfully')
print(X)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('Dataset split into training and test sets successfully')
print('X_train:', X_train)

#Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print('Multiple Linear Regression model trained successfully')

#Predicting the Test set results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#Visualising the Training set results
plt.scatter(regressor.predict(X_train), Y_train, color='red')
plt.plot(regressor.predict(X_train), regressor.predict(X_train), color='blue')
plt.title('Truth or Bluff (Training set)')
plt.xlabel('Predicted Salary')
plt.ylabel('Actual Salary')
plt.show()

#Visualising the Test set results
plt.scatter(regressor.predict(X_test), Y_test, color='red')
plt.plot(regressor.predict(X_test), regressor.predict(X_test), color='blue')
plt.title('Truth or Bluff (Test set)')
plt.xlabel('Predicted Salary')
plt.ylabel('Actual Salary')
plt.show()