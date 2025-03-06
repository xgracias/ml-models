# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_prediction = regressor.predict(X_test)

# visualising the training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (Training set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualising the test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()