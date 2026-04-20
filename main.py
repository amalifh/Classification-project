import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils

#Constant
N_CLASSES = 3
N_FEATURES = 4
TRAINING_SIZE = 30
TESTING_SIZE = 20
ALPHA = 0.01
ITERATIONS = 1000

#DATAFRAMES
df1 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "class_1")
df2 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "class_2")
df3 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "class_3")

#training
df = pd.concat([df1[:TRAINING_SIZE], df2[:TRAINING_SIZE], df3[:TRAINING_SIZE]], ignore_index=True)
X_train = df.iloc[:, :4].to_numpy(dtype=float)
#testing
df = pd.concat([df1[-TESTING_SIZE:], df2[-TESTING_SIZE:], df3[-TESTING_SIZE:]], ignore_index=True)
X_test = df.iloc[:, :4].to_numpy(dtype=float)


def main():
    #Task 1 - Part one
    W = np.zeros((N_CLASSES, N_FEATURES + 1), dtype=float) 
    W, loss = utils.train_classifier(X_train, W, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)
#    print(loss[:20])
#    print(loss[-20:])
    cm_training = utils.confusion_matrix(W, X_train, TRAINING_SIZE, N_CLASSES)
    cm_testing = utils.confusion_matrix(W, X_test, TESTING_SIZE, N_CLASSES)
    print(cm_training)
    print(cm_testing)
if __name__ == '__main__':
    main()  
