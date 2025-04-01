# Importing the necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Load the dataset
dataset = pd.read_csv("Data.csv", na_values=["", "?", "-"])

# Identify missing data (assumes that missing data is represented as NaN)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Print the number of missing entries in each column
print("Datos antes de imputar:")
print(X)
print(Y)

# Configure an instance of the SimpleImputer class
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(X[:, 1:3])

# Apply the transform to the DataFrame
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Print your updated matrix of features
print("Datos después de imputar:")
print(X[:, 1:3])

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("Datos categóricos después de encodificar:")
print(X)

#Encoding dependent data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print("Datos dependientes después de encodificar:")
print(Y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)