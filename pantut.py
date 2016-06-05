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


