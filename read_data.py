import numpy as np

#klassene inneholder infoen: sepal length, sepal width, petal length, petal width
def create_set(N, file):
    with open(file, 'r') as f:
        data = f.readlines()
        print(data)

create_set(1, "class_1")