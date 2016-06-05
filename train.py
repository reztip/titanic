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

# ------------------------------------------------
# start the 

fare_ceiling = 40
data[data[:, 9].astype(np.float) >= fare_ceiling , 9] = fare_ceiling - 1.0
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling//fare_bracket_size

#correponds to classes 1,2,3 on board
number_of_classes = 3
#This is superior
number_of_classes = len(np.unique(data[:,2]))

# initialize with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets)) 

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        women_only_stats = data[ \
        (data[:,4] == 'female') \
        &(data[:,2].astype(np.float) == i+1) \
        &(data[:,9].astype(np.float) >= j * fare_bracket_size) \
        &(data[:,9].astype(np.float) < (j+1) * fare_bracket_size) \
        , 1]

        men_only_stats = data[ \
        (data[:,4] != 'female') \
        &(data[:,2].astype(np.float) == i+1) \
        &(data[:,9].astype(np.float) >= j * fare_bracket_size) \
        &(data[:,9].astype(np.float) < (j+1) * fare_bracket_size) \
        , 1]

survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

survival_table[ survival_table != survival_table ] = 0
survival_table[survival_table < .5 ] = 0
survival_table[survival_table >= .5 ] = 1
test_file = open('test.csv', 'rt')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
predictions_file = open("genderclassmodel.csv", "wt")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets - 1
            break
        if row[8] >= j * fare_bracket_size and \
           row[8]  < (j + 1) * fare_bracket_size:
           bin_fare = j
           break

