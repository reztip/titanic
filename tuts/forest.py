# This particular tutorial is about using random forest classifiers
import csv as csv
import numpy as np
import pandas as pd


csv_file_object = csv.reader(open('train.csv', 'rt'))
header = csv_file_object.__next__()


