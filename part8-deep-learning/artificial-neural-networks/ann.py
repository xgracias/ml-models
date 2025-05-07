import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# encoding categorical data
## label encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

## one hot encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANNNNNNNNNNNNNNNNNNN
## initialize the ANN
ann = tf.keras.models.Sequential()
## add the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
## add the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
## add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train the ANNNNNNNNNNNNNNNNNNNNNNNNN
## compile the ANN
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
## train the ANN
ann.fit(X_train, y_train, batch_size = 32, epochs=100)

# predict one
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# predict test results
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# make the confusing matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)