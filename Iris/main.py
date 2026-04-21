import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_iris
import Iris.utils as utils

#CONSTANTS
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

df_train = pd.concat([df1[:TRAINING_SIZE], df2[:TRAINING_SIZE], df3[:TRAINING_SIZE]], ignore_index=True)
X_train = df_train.iloc[:, :4].to_numpy(dtype=float)

df_test = pd.concat([df1[-TESTING_SIZE:], df2[-TESTING_SIZE:], df3[-TESTING_SIZE:]], ignore_index=True)
X_test = df_test.iloc[:, :4].to_numpy(dtype=float)

iris = load_iris()

def main():
    #Task 1 - Part one
    W = np.zeros((N_CLASSES, N_FEATURES + 1), dtype=float) 
    W, loss = utils.train_classifier(X_train, W, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)

    cm_training = utils.confusion_matrix(W, X_train, TRAINING_SIZE, N_CLASSES)
    cm_testing = utils.confusion_matrix(W, X_test, TESTING_SIZE, N_CLASSES)
    error_rate_training = utils.error_rate(cm_training)
    error_rate_testing = utils.error_rate(cm_testing)

    #Task 2 - Part two
    #Removing one feature
    X_3_train= utils.remove_feature(X_train, 0)
    X_3_test = utils.remove_feature(X_test, 0)

    #Removing 2 features
    X_2_train = utils.remove_feature(X_3_train, 0)
    X_2_test = utils.remove_feature(X_3_test, 0)


    #Removing 3 features
    X_1_train = utils.remove_feature(X_2_train, 0)
    X_1_test = utils.remove_feature(X_2_test, 0)

    #Training for removed features
    W_3 = np.zeros((N_CLASSES, N_FEATURES))
    W_3feat, loss_3 = utils.train_classifier(X_3_train, W_3, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)
    cm_3_train = utils.confusion_matrix(W_3feat, X_3_train, TRAINING_SIZE, N_CLASSES)
    cm_3_test = utils.confusion_matrix(W_3feat, X_3_test, TESTING_SIZE, N_CLASSES)
    error_rate_3_train = utils.error_rate(cm_3_train)
    error_rate_3_test = utils.error_rate(cm_3_test)

    W_2 = np.zeros((N_CLASSES, N_FEATURES-1))
    W_2feat, loss_2 = utils.train_classifier(X_2_train, W_2, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)
    cm_2_train = utils.confusion_matrix(W_2feat, X_2_train, TRAINING_SIZE, N_CLASSES)
    cm_2_test = utils.confusion_matrix(W_2feat, X_2_test, TESTING_SIZE, N_CLASSES)
    error_rate_2_train = utils.error_rate(cm_2_train)
    error_rate_2_test = utils.error_rate(cm_2_test)
    
    W_1 = np.zeros((N_CLASSES, N_FEATURES-2))
    W_1, loss_1 = utils.train_classifier(X_1_train, W_1, TRAINING_SIZE, ITERATIONS, ALPHA, N_CLASSES)
    cm_1_train = utils.confusion_matrix(W_1, X_1_train, TRAINING_SIZE, N_CLASSES)
    cm_1_test = utils.confusion_matrix(W_1, X_1_test, TESTING_SIZE, N_CLASSES)
    error_rate_1_train = utils.error_rate(cm_1_train)
    error_rate_1_test = utils.error_rate(cm_1_test)

    plt.plot(loss_1)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Convergence")
    plt.savefig("Iris/figures/loss_1", dpi = 500)


    disp_training = ConfusionMatrixDisplay(confusion_matrix=cm_1_train, display_labels=iris.target_names)
    disp_training.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig("Iris/figures/CM_1_train", dpi = 500)


    disp_training = ConfusionMatrixDisplay(confusion_matrix=cm_1_test, display_labels=iris.target_names)
    disp_training.plot(cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.savefig("Iris/figures/CM_1_test", dpi = 500)
    
    print(f'Error rate training, all: {error_rate_training}')
    print(f'Error rate testing, all: {error_rate_testing}')

    print(f'Error rate training, 3 features: {error_rate_3_train}')
    print(f'Error rate testing, 3 features: {error_rate_3_test}')

    print(f'Error rate training, 2 features: {error_rate_2_train}')
    print(f'Error rate testing, 2 features: {error_rate_2_test}')

    print(f'Error rate training, 1 feature: {error_rate_1_train}')
    print(f'Error rate testing, 1 feature: {error_rate_1_test}')

    features = [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    for i, ax in enumerate(axes.flat):
        ax.hist(df1.iloc[:,i], bins=15, alpha=0.5, label='Class 1')
        ax.hist(df2.iloc[:,i], bins=15, alpha=0.5, label='Class 2')
        ax.hist(df3.iloc[:,i], bins=15, alpha=0.5, label='Class 3')

        ax.set_title(features[i])
        ax.set_xlabel(features[i])
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig("figures/histogram", dpi = 500)
    plt.show()

"""    print(f'Error rate training: {error_rate_training}')
    print(f'Error rate testing: {error_rate_testing}')
    
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Convergence")
    plt.show()

    disp_training = ConfusionMatrixDisplay(confusion_matrix=cm_training, display_labels=iris.target_names)
    disp_training.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for training')
    plt.show()

    disp_testing = ConfusionMatrixDisplay(confusion_matrix=cm_testing, display_labels=iris.target_names)
    disp_testing.plot(cmap=plt.cm.Greens)
    plt.title('Confusion Matrix for testing')
    plt.show()"""
"""#----------------------------PART TWO----------------------------
    features = [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    for i, ax in enumerate(axes.flat):
        ax.hist(df1.iloc[:,i], bins=15, alpha=0.5, label='Class 1')
        ax.hist(df2.iloc[:,i], bins=15, alpha=0.5, label='Class 2')
        ax.hist(df3.iloc[:,i], bins=15, alpha=0.5, label='Class 3')

        ax.set_title(features[i])
        ax.set_xlabel(features[i])
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.show()

"""

if __name__ == '__main__':
    main()  
