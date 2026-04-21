import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_iris
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

iris = load_iris()

def main():
    #Task 1 - Part one
    W = np.zeros((N_CLASSES, N_FEATURES + 1), dtype=float) 
    W, loss = utils.train_classifier(X_train, W, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)
#    print(loss[:20])
#    print(loss[-20:])
    cm_training = utils.confusion_matrix(W, X_train, TRAINING_SIZE, N_CLASSES)
    cm_testing = utils.confusion_matrix(W, X_test, TESTING_SIZE, N_CLASSES)
    error_rate_training = utils.error_rate(cm_training)
    error_rate_testing = utils.error_rate(cm_testing)
    print(f'Error rate training: {error_rate_training}')
    print(f'Error rate testing: {error_rate_testing}')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_training, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for training')
    plt.show()

    disp_testing = ConfusionMatrixDisplay(confusion_matrix=cm_testing, display_labels=iris.target_names)
    disp_testing.plot(cmap=plt.cm.Greens)
    plt.title('Confusion Matrix for testing')
    plt.show()


if __name__ == '__main__':
    main()  
