import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_iris
import utils

#CONSTANTS
N_CLASSES = 3
N_FEATURES = 4
TRAINING_SIZE = 30
TESTING_SIZE = 20
ALPHAs = [0.01, 0.05, 0.001, 0.005]
STEP_FACTOR = 0.01
ITERATIONS = 1000

#DATAFRAMES
df1 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "Iris\class_1")
df2 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "Iris\class_2")
df3 = utils.get_set(TRAINING_SIZE + TESTING_SIZE, "Iris\class_3")

df_train = pd.concat([df1[:TRAINING_SIZE], df2[:TRAINING_SIZE], df3[:TRAINING_SIZE]], ignore_index=True)
X_train = df_train.iloc[:, :4].to_numpy(dtype=float)

df_test = pd.concat([df1[-TESTING_SIZE:], df2[-TESTING_SIZE:], df3[-TESTING_SIZE:]], ignore_index=True)
X_test = df_test.iloc[:, :4].to_numpy(dtype=float)

iris = load_iris()

def main():
#-----------------------TASK ONE-----------------------
    #Testing for different step factors to find the optimal one   
    all_losses = {}
    for alpha in ALPHAs:
        W = np.zeros((N_CLASSES, N_FEATURES + 1), dtype=float) 

        W, loss = utils.train_classifier(X_train, W, TRAINING_SIZE, ITERATIONS, alpha, N_CLASSES)
        
        all_losses[alpha] = loss
        
        cm_training = utils.confusion_matrix(W, X_train, TRAINING_SIZE, N_CLASSES)
        cm_testing = utils.confusion_matrix(W, X_test, TESTING_SIZE, N_CLASSES)
        
        error_rate_training = utils.error_rate(cm_training)
        error_rate_testing = utils.error_rate(cm_testing)    

    #Getting confusion matrix and error rate for chosen step factor
    W = np.zeros((N_CLASSES, N_FEATURES + 1))
    W, loss = utils.train_classifier(X_train, W, TRAINING_SIZE, ITERATIONS, STEP_FACTOR, N_CLASSES)

    cm_training = utils.confusion_matrix(W, X_train, TRAINING_SIZE, N_CLASSES)
    cm_testing = utils.confusion_matrix(W, X_test, TESTING_SIZE, N_CLASSES)
    error_rate_training = utils.error_rate(cm_training)
    error_rate_testing = utils.error_rate(cm_testing)
    print(f'Error rate training {error_rate_training}')
    print(f'Error rate testing {error_rate_testing}')

#-----------------------TASK TWO-----------------------
    #Removing one feature, sepal length
    X_3_train= utils.remove_feature(X_train, 0)
    X_3_test = utils.remove_feature(X_test, 0)

    #Removing 2 features, sepal width and length
    X_2_train = utils.remove_feature(X_3_train, 0)
    X_2_test = utils.remove_feature(X_3_test, 0)


    #Removing 3 features, sepal width and length and petal length
    X_1_train = utils.remove_feature(X_2_train, 0)
    X_1_test = utils.remove_feature(X_2_test, 0)

    #Training and testing for all the removed features
    W_3 = np.zeros((N_CLASSES, N_FEATURES))
    W_3feat, loss_3 = utils.train_classifier(X_3_train, W_3, TRAINING_SIZE, ITERATIONS, STEP_FACTOR, N_CLASSES)
    cm_3_train = utils.confusion_matrix(W_3feat, X_3_train, TRAINING_SIZE, N_CLASSES)
    cm_3_test = utils.confusion_matrix(W_3feat, X_3_test, TESTING_SIZE, N_CLASSES)
    error_rate_3_train = utils.error_rate(cm_3_train)
    error_rate_3_test = utils.error_rate(cm_3_test)

    W_2 = np.zeros((N_CLASSES, N_FEATURES-1))
    W_2feat, loss_2 = utils.train_classifier(X_2_train, W_2, TRAINING_SIZE, ITERATIONS, STEP_FACTOR, N_CLASSES)
    cm_2_train = utils.confusion_matrix(W_2feat, X_2_train, TRAINING_SIZE, N_CLASSES)
    cm_2_test = utils.confusion_matrix(W_2feat, X_2_test, TESTING_SIZE, N_CLASSES)
    error_rate_2_train = utils.error_rate(cm_2_train)
    error_rate_2_test = utils.error_rate(cm_2_test)

    W_1 = np.zeros((N_CLASSES, N_FEATURES-2))
    W_1, loss_1 = utils.train_classifier(X_1_train, W_1, TRAINING_SIZE, ITERATIONS, STEP_FACTOR, N_CLASSES)
    cm_1_train = utils.confusion_matrix(W_1, X_1_train, TRAINING_SIZE, N_CLASSES)
    cm_1_test = utils.confusion_matrix(W_1, X_1_test, TESTING_SIZE, N_CLASSES)
    error_rate_1_train = utils.error_rate(cm_1_train)
    error_rate_1_test = utils.error_rate(cm_1_test)

    print(f'Error rate training, all: {error_rate_training}')
    print(f'Error rate testing, all: {error_rate_testing}')

    print(f'Error rate training, 3 features: {error_rate_3_train}')
    print(f'Error rate testing, 3 features: {error_rate_3_test}')

    print(f'Error rate training, 2 features: {error_rate_2_train}')
    print(f'Error rate testing, 2 features: {error_rate_2_test}')

    print(f'Error rate training, 1 feature: {error_rate_1_train}')
    print(f'Error rate testing, 1 feature: {error_rate_1_test}')

    plt.figure()

    plt.plot(loss_3, label='3 features')
    plt.plot(loss_2, label='2 features')
    plt.plot(loss_1, label='1 feature')
    plt.plot(loss, label = 'All features' )

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over iterations for different feature counts')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss_comparison', dpi = 500)

if __name__ == '__main__':
    main()  
