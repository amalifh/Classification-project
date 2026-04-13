import numpy as np
import os

#klassene inneholder infoen: sepal length, sepal width, petal length, petal width
def create_set(size, file, name_set):

    with open(file, 'r')as f:
        raw_data = np.array(f.readlines())
        new_set = []
        
        for i, line in enumerate(raw_data[:size]):
            data = line.strip()
            new_set.append(data)
    f.close()
    with open(name_set, 'w+') as f:
        for element in new_set:
            f.write('%s\n' %element)

create_set(30, "class_1", "class1_training")
create_set(20, 'class_1', 'class1_testing')
create_set(30, 'class_2', 'class2_training'),
create_set(20, 'class_2', 'class2_testing')
create_set(30, 'class_3', 'class3_training')
create_set(20, 'class_3', 'class3_testing')
