import numpy as np
import csv
import json
import os

#klassene inneholder infoen: sepal length, sepal width, petal length, petal width
def create_set(size, file, name_set):

    with open(file, 'r')as f:
        raw_data = np.array(f.readlines())
        new_set = []

        for i, line in enumerate(raw_data[:size]):
            data = line.strip()
            new_set.append(data)
    with open(name_set, 'w+') as f:
        for element in new_set:
            f.write('%s\n' %element)


3
def create_set(file, size, name, type):
    
    with open(file, 'r') as f:
        new_set = {}
        data_set = []
        raw_data = np.array(f.readlines())
        for line in raw_data[:size]:
            data = line.strip()
            clean_data= []
            for element in data:
                clean_data.append(float(element))

            data_set.append(clean_data)

    new_set = { 
        "Class" : type,
        "data" : data_set
    }

    with open(name, 'w') as f:
        json.dump(new_set, f)

def create_set4(file, size, name, type_):
    data_set = []

    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i >= size:
                break
            clean_data = [float(x) for x in line.strip().split(',')]
            data_set.append(clean_data)

    new_set = {
        "Class": type_,
        "data": data_set
    }

    with open(name, 'w') as f:
        json.dump(new_set, f, indent=4)

def create_set2(file, size, name, class_label):
    with open(file, 'r') as f_in, open(name, 'w', newline='') as f_out:
        writer = csv.writer(f_out)

        for i, line in enumerate(f_in):
            if i >= size:
                break

            # split numbers (change ',' if needed)
            row = [float(x) for x in line.strip().split(',')]
            
            # append class label
            row.append(class_label)

            writer.writerow(row)

create_set2("class_1", 30, "class1_training", "Setosa")
create_set2('class_1', 20, 'class1_testing', 'Setosa')
create_set2('class_2', 30, 'class2_training', 'Versicolor'),
create_set2('class_2', 20,  'class2_testing', 'Versicolor')
create_set2('class_3', 30, 'class3_training', 'Virginica' )
create_set2('class_3', 20, 'class3_testing', 'Virginica')