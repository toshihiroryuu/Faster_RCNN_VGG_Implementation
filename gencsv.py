
import csv

PATH='/home/quest'

f1 = open('/home/quest/train.csv')
csv_f = csv.reader(f1)





for row in csv_f:
  name=row[0]
  listt=''
  listt='/home/quest/udacity_driving_datasets/'+name
  with open('/home/quest/ttrraaiinn.csv', mode='a') as csv_file:
    fieldnames = ['file_path', 'x1', 'y1','x2', 'y2', 'class_id']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # writer.writeheader()
    writer.writerow({'file_path': listt, 'x1': float(row[2])/2, 'y1': float(row[4])/2, 'x2': float(row[1])/2, 'y2': float(row[3])/2, 'class_id':row[5]})
    print("converted xmin, xmax, ymin, ymax into x1, y1, x2, y2")
    print("copied row")
    
 
  




