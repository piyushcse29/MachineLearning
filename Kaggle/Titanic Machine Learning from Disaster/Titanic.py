# Kaggle: Titanic Machine Learning from Disaster
# Author : Piyush Mittal

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# Drop feature ticket, cabin and passengerId
X_train = X_train.drop(['Ticket', 'Cabin','PassengerId'], axis=1)

#Making a new column based on title of Name
X_train['Title'] = X_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


#Name Title correction
X_train['Title'] = X_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Uncommon')
X_train['Title'] = X_train['Title'].replace('Mlle', 'Miss')
X_train['Title'] = X_train['Title'].replace('Ms', 'Miss')
X_train['Title'] = X_train['Title'].replace('Mme', 'Mrs')

# Drop Name feature
X_train = X_train.drop(['Name'], axis=1)

#Split to y_train
y_train = X_train.iloc[:, 0].values
X_train = X_train.drop(['Survived'], axis=1)


#Mapping Sex to numerical values
X_train['Sex'] = X_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
 
guess_ages = np.zeros((2,3)) 
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = X_train[(X_train['Sex'] == i) & (X_train['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
        for j in range(0, 3):
            X_train.loc[ (X_train.Age.isnull()) & (X_train.Sex == i) & (X_train.Pclass == j+1),'Age'] = guess_ages[i,j]

X_train['Age'] = X_train['Age'].astype(int)        
 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()