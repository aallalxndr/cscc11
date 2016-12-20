from knn import euclidenDistance
from knn import getNeighbour
from knn import getResponse
from knn import correctness
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#read the files
train = pd.read_csv("C:\Users/aalla/Documents/cscc11/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("C:\Users/aalla/Documents/cscc11/test.csv", dtype={"Age": np.float64}, )

#merge data
titanic = pd.merge(train,test, how='outer')



coltitle = titanic['Name'].apply(lambda s: pd.Series({'Title': s.split(',')[1].split('.')[0].strip(),
                                                      'LastName':s.split(',')[0].strip(), 'FirstName':s.split(',')[1].split('.')[1].strip()}))
# Add the columns to the titanic dataframe
titanic = pd.concat([titanic, coltitle], axis=1) 
# Drop the Name column
titanic.drop('Name', axis=1, inplace=True)

# Also reassign mlle, ms, and mme accordingly
titanic.loc[titanic['Title']=='Mlle', 'Title']='Miss'.strip()
titanic.loc[titanic['Title']=='Ms', 'Title']='Miss'.strip()
titanic.loc[titanic['Title']=='Mme', 'Title']='Mrs'.strip()

# Get the count of female and male passengers based on titles
tab = titanic.groupby(['Sex', 'Title']).size()

# Total number of families
titanic['total_members'] = titanic.SibSp + titanic.Parch + 1

# Drop the Ticket and Cabin column 
titanic.drop('Cabin', axis=1, inplace=True)
titanic.drop('Ticket', axis=1, inplace=True)

titanic[['Pclass', 'Fare']].groupby('Pclass').mean()
titanic.loc[titanic.PassengerId==1044.0, 'Fare']=13.30

titanic[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean()
# Also lets try to find the fare based on Embarked 
titanic[['Embarked', 'Fare']].groupby('Embarked').mean()
titanic.loc[titanic['Embarked'].isnull() == True, 'Embarked']='C'.strip()


pd.pivot_table(titanic, index=['Sex', 'Title', 'Pclass'], values=['Age'], aggfunc='median')

# Missing values of age
# a function that fills the missing values of the Age variable
def fillAges(row):
    
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 29.5
        elif row['Title'] == 'Mrs':
            return 38.0
        elif row['Title'] == 'Dr':
            return 49.0
        elif row['Title'] == 'Lady':
            return 48.0
        elif row['Title'] == 'the Countess':
            return 33.0

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return 24.0
        elif row['Title'] == 'Mrs':
            return 32.0

    elif row['Sex']=='female' and row['Pclass'] == 3:
        
        if row['Title'] == 'Miss':
            return 9.0
        elif row['Title'] == 'Mrs':
            return 29.0

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return 4.0
        elif row['Title'] == 'Mr':
            return 36.0
        elif row['Title'] == 'Sir':
            return 49.0
        elif row['Title'] == 'Capt':
            return 70.0
        elif row['Title'] == 'Col':
            return 58.0
        elif row['Title'] == 'Don':
            return 40.0
        elif row['Title'] == 'Dr':
            return 38.0
        elif row['Title'] == 'Major':
            return 48.5

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return 1.0
        elif row['Title'] == 'Mr':
            return 30.0
        elif row['Title'] == 'Dr':
            return 38.5

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return 4.0
        elif row['Title'] == 'Mr':
            return 22.0


titanic['Age'] = titanic.apply(lambda s: fillAges(s) if np.isnan(s['Age']) else s['Age'], axis=1)

# Convert sex to 0 and 1 (Female and Male)
def trans_sex(x):
    if x == 'female':
        return 0
    else:
        return 1
titanic['Sex'] = titanic['Sex'].apply(trans_sex)

# Convert Embarked to 1, 2, 3 (S, C, Q)
def trans_embark(x):
    if x == 'S':
        return 3
    if x == 'C':
        return 2
    if x == 'Q':
        return 1
titanic['Embarked'] = titanic['Embarked'].apply(trans_embark)    

# Add a child and mother column for predicting survivals
titanic['Child'] = 0
titanic.loc[titanic['Age']<18.0, 'Child'] = 1
titanic['Mother'] = 0
titanic.loc[(titanic['Age']>18.0) & (titanic['Parch'] > 0.0) & (titanic['Sex']==0) & (titanic['Title']!='Miss'), 'Mother'] =1

features_label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'total_members', 'Child', 'Mother']
target_label= ['Survived']
train = titanic[titanic['Survived'].isnull()!= True]
test = titanic[titanic['Survived'].isnull()== True]


def main():
    
    predictions = []
    k = 3
    for x in range(len(test)):
        neighbours = getNeighbour(train,test[x], k)
        result = getResponse(neighbours)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(test, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
main()