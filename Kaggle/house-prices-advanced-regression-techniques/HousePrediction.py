#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:80].values

y = dataset.iloc[:, 80].values

#Spliting the dataset into train set and test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#FeatureScaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X)



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 1]= labelEncoder_X.fit_transform(X[:, 1])
X[:, 78]= labelEncoder_X.fit_transform(X[:, 78])
X[:, 77]= labelEncoder_X.fit_transform(X[:, 77])

X[:, 75]= labelEncoder_X.fit_transform(X[:, 75])
X[:, 5]= labelEncoder_X.fit_transform(X[:, 5])


onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X)


#Fitting Simple Linear Regression to the trainig set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience training set')
plt.show()

#Visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience test set')
plt.show()