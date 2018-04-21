# Kaggle: Titanic Machine Learning from Disaster
# Author : Piyush Mittal

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')



def prapare_data(dataset):
    

    # Drop feature ticket, cabin and passengerId
    dataset = dataset.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
    
    #Making a new column based on title of Name
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    
    #Name Title correction
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Uncommon')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Drop Name feature
    dataset = dataset.drop(['Name'], axis=1) 
    
    #Mapping Sex to numerical values
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
     
    guess_ages = np.zeros((2,3)) 
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
    
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    
    for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
    
    dataset['Age'] = dataset['Age'].astype(int)        
    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
        
    dataset = dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
    #Filling missing embarked values with mode
    freq_port = dataset.Embarked.dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    #Filling fare missing values with median
    
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    
    
    #Creating 4 bins for fare
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset = dataset.drop(['FareBin'], axis=1)
    
    return dataset


X_train = prapare_data(dataset_train)
X_test =  prapare_data(dataset_test)

#Split to y_train and y_test
y_train = X_train.iloc[:, 0].values
X_train = X_train.drop(['Survived'], axis=1)

 
 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


my_kaggle_submission = pd.DataFrame({
    'PassengerId':dataset_test['PassengerId'],
    'Survived':y_pred
})

my_kaggle_submission.to_csv('csv_to_submit.csv', index = False)