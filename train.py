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
# Check who is female and alive: what is proportion
female_and_survived = np.logical_and(survived, females_bool)
proportion_women_survived = np.sum(female_and_survived) / np.sum(females_bool)



