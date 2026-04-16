import pandas as pd

#klassene inneholder infoen: sepal length, sepal width, petal length, petal width
def create_set(size, file, name_new, type_class):
    cols = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'iris_type']
    df = pd.read_csv(file,nrows=size, names=cols)
    df['iris_type'] = type_class
    df.to_csv(name_new, columns=cols, index=False)

create_set(30, 'class_1', 'class_1_training', 'Setosa')
create_set(20,'class_1', 'class_1_testing', 'Setosa')
create_set(30, 'class_2', 'class_2_testing', 'Versicolor')
create_set(20, 'class_2', 'class_2_training', 'Versicolor')
create_set(30, 'class_3', 'class_3_training', 'Virginica')
create_set(20, 'class_3', 'class_3_testing', 'Versicolor')
