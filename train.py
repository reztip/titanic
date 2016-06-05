# Import relevant packages
import csv as csv
import numpy as np

# Open up a csv file as a python object

csv_file_object = csv.reader(open('train.csv', 'rt', encoding= 'utf8'))
header = csv_file_object.__next__()

data = []
for row in csv_file_object:
    data.append(row)

data = np.array(data)

# line does not work below
# data[:, 0] = data[:, 0].astype(np.int)

survived = data[:,1] == '1'
number_passengers = np.size(data[:, 0])
number_survived = np.sum(survived)

females_bool = data[:,4] == 'female'
males_bool = data[:,4] != 'female'

# Check who is female/male and alive: what is proportion
female_and_survived = np.logical_and(survived, females_bool)
male_and_survived = np.logical_and(survived, males_bool)
# How many survived
proportion_women_survived = np.sum(female_and_survived) / np.sum(females_bool)
proportion_men_survived = np.sum(male_and_survived) / np.sum(males_bool)

test_file = open('test.csv', 'rt')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

prediction_file = open("gender_based_model.csv", 'wt')
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(['PassengerId', 'Survived'])
# We predict women survive, men do not
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], '1'])
    else:
        prediction_file_object.writerow([row[0], '0'])

# close the files
test_file.close()
prediction_file.close()


