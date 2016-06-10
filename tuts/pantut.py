# This particular tutorial is about using the pandas lib
import csv as csv
import numpy as np
import pandas as pd

csv_file_object = csv.reader(open('train.csv', 'rt'))
header = csv_file_object.__next__()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

df = pd.read_csv('train.csv', header = 0) 

df['Gender'] =  df['Sex'].map( {'female': 0, 'male': 1}).astype(int)
embarked_na_fill = 'X'
df['emb'] = df['Embarked'].fillna(embarked_na_fill).map( \
        {'S': 0, 'C': 1,'Q': 2, embarked_na_fill: 3} ).astype(int)

median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
         (df['Pclass'] == j+1)]['Age'].dropna().median()

# Feature generation
df['AgeFill'] = df['Age']
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']

# Fill in missing age data with medians by class and gender
for i in range(0,2):
    for j in range(0,3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
         'AgeFill'] = median_ages[i,j]

df['AgexClass'] = df['AgeFill'] * df['Pclass']
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
df = df.drop(['Age'], axis = 1)
df = df.dropna()

train_data = df.values

