import pandas as pd
import numpy as np
import matplotlib.pyplot as P
import math
import csv
np.random.seed(1) # Set seed as 1 for monitoring performance
from sklearn.ensemble import AdaBoostClassifier

"""REMEMBER THE PIPELINE
0) Randomize data and split into train/CV sets
1) Feature Engineering/Extraction
2) Feature Standardization & mean normalization
3) Choosing Classifiers  - probably svm or logistic classifs
5) Rinse & Repeat until satisfied
"""

def shuffle(dframe, n = 1, axis = 0):
    dframe = dframe.copy()
    for x in range(n):
        dframe.apply(np.random.shuffle, axis = axis)
    return dframe

# read in the dataset - split into train and cross_validation
# Choose 80% for train, 20% for crossvalidation
df = pd.read_csv('train.csv', header = 0)
test_df = pd.read_csv('test.csv', header = 0)
# do a random shuffle so your classif is not too biased
df = shuffle(df)
length = df.shape[0]
cutoff = math.floor(length * .8)
cross_validation = df[cutoff:]
df = df[:cutoff]

# Preprocess existing features: 
#------------------------------------------------------------
# fill missing age value with median
df['Age'] = df.Age.fillna(df.Age.median())
test_df['Age'] = test_df.Age.fillna(test_df.Age.median())

#------------------------------------------------------------
# Create new feature: age group.
# Currently have 8 groups
n_age_buckets = 6
df['AgeCategory'] = pd.qcut(df.Age, q = n_age_buckets, labels = False)
test_df['AgeCategory'] = pd.qcut(test_df.Age, q = n_age_buckets, labels = False)

# Create new feature: Gender = integer sex - 1 female, 0 male

df['Gender'] = 0
test_df['Gender'] = 0
df.Gender[df.Sex == 'female'] = 1
test_df.Gender[test_df.Sex == 'female'] = 1

# I choose C because C-embarkers are more likely to survive in
# the data, and those with missing Embarkations did survive
df.Embarked = df.Embarked.fillna('C')
test_df.Embarked = test_df.Embarked.fillna('C')
df['Embark'] = 0
test_df['Embark'] = 0
df.Embark[df.Embarked == 'Q'] = 1
test_df.Embark[test_df.Embarked == 'Q'] = 1
df.Embark[df.Embarked == 'S'] = 2
test_df.Embark[test_df.Embarked == 'S'] = 2

# Create a family feature - no family means little survivability
# A large amount of family means little survivablility
# Some family means you probably survived
df['Family'] = df.SibSp + df.Parch
test_df['Family'] = test_df.SibSp + test_df.Parch

df['FamClass'] = df.Family * df.Pclass
test_df['FamClass'] = test_df.Family * test_df.Pclass
# Arbitrarily make 30 to fit model better
df['FamClass'][df['Family'] == 0] = 30
test_df['FamClass'][test_df['Family'] == 0] = 30

#New cfeature: category for fare
df['FareCat'] = pd.cut(df.Fare, bins = 5, labels = False)
test_df['FareCat'] = pd.cut(test_df.Fare, bins = 5, labels = False)
#------------------------------------------------------------

# Perform Feature normalization

df.Age = (df.Age - df.Age.mean())/df.Age.std()
test_df.Age = (test_df.Age - test_df.Age.mean())/test_df.Age.std()
df.Fare = (df.Fare - df.Fare.mean())/df.Fare.std()
test_df.Fare = (test_df.Fare - test_df.Fare.mean())/test_df.Fare.std()
df.FamClass = (df.FamClass ) / df.FamClass.std()
test_df.FamClass = (test_df.FamClass ) / test_df.FamClass.std()

df['AgeSquared'] = df.Age * df.Age
test_df['AgeSquared'] = test_df.Age * test_df.Age
df['AgeSquaredCategory'] = pd.cut(df.AgeSquared, bins  = n_age_buckets, labels = False)
test_df['AgeSquaredCategory'] = pd.cut(test_df.AgeSquared, bins  = n_age_buckets, labels = False)
df['FamClassGender'] = df.FamClass * df.Gender
test_df['FamClassGender'] = test_df.FamClass * test_df.Gender

# drop features not being used - play with feature selection here
#------------------------------------------------------------

train_df = df.drop(['Age', 'Sex', 'Cabin', 'Ticket', 'Embarked', 'Name', 'Survived', 'SibSp', 'Parch', 'Fare', 'Family', 'AgeCategory'], axis = 1)
test_df = test_df.drop(['Age', 'Sex', 'Cabin', 'Ticket', 'Embarked', 'Name',  'SibSp', 'Parch', 'Fare', 'Family', 'AgeCategory'], axis = 1)
ids = test_df['PassengerId'].values
test_df = test_df.drop(['PassengerId'], axis = 1)

#------------------------------------------------------------
# Convert features to numpy array
train_data = train_df.values
test_data = test_df.values
#------------------------------------------------------------
#Training and prediction

print('Training')
classif = AdaBoostClassifier(n_estimators = 100)
classif.fit(train_data[:, 1:], train_data[:,0])

print('Predicting')
#------------------------------------------------------------

output = classif.predict(test_data).astype(int)
predictions_file = open('predictions.csv', 'wt')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done')

