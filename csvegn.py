
import csv

f = open('/home/quest/train.csv',"w")
csv_f = csv.writer(f)
PATH='/home/quest/udacity_driving_datasets/'

for row in csv_f:
  row[0].append(PATH)

  