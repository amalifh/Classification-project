import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

mama = 4
#klassene inneholder infoen: sepal length, sepal width, petal length, petal width
def get_set(size, file):
    cols = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
    df = pd.read_csv(file,nrows=size, names=cols)
    return df

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#vi skal trene hele klassen samtidig, lager da en df med alle 90 trainings
#the W_matrix will contain the bias
#her skal ikke x inneholde target value
def confidence_vector(W_matrix, x_features, batch_size, num_classes):
    g_all = np.zeros(((batch_size * num_classes), num_classes))
    for i, x in enumerate(x_features):
        x = np.append(x, 1)
        z = np.matmul(W_matrix, x)
        g_all[i] = sigmoid(z)
    return g_all

def MSE_loss(g, t):
    error = g - t
    loss = 0.5 * np.sum(error**2)
    return loss

def MSE_gradient(g, t, x, num_features, num_classes):
    grad = np.zeros((num_classes, num_features + 1))

    for x_k, g_k, t_k in zip(x, g, t):
        x_k = np.append(x_k, 1)
        x_k = x_k.reshape(num_features +1, 1)
        error = (g_k - t_k).reshape(num_classes, 1)
        sigmoid_derivative = (g_k * (1 - g_k)).reshape(num_classes, 1)

        delta = error * sigmoid_derivative
        grad += delta @ np.transpose(x_k)
    return grad

def train_classifier(training_data, W, training_size, iterations, alpha, num_classes):
    target1 = np.tile([1,0,0], training_size)
    target2 = np.tile([0,1,0], training_size)
    target3 = np.tile([0,0,1], training_size)

    target = np.concatenate((target1, target2, target3), axis=None)
    target = np.reshape(target, (training_size * num_classes, num_classes))

    loss = np.zeros(iterations, dtype=float)

    for i in range(iterations):
        g_vector = confidence_vector(W, training_data, training_size, num_classes)
        gradient = MSE_gradient(g_vector, target, training_data, len(training_data[0]), num_classes)

        loss[i] = MSE_loss(g_vector, target)
        W = W - alpha * gradient

        print(f"Iteration: {i+1}")

    return W, loss

def confusion_matrix(W, data, batch_size, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    predictions = confidence_vector(W, data, batch_size, num_classes)
    row = -1
    for i, g_k in enumerate(predictions):
        if i % batch_size == 0:
            row += 1
        col = np.argmax(g_k)
        matrix[row][col] += 1
    
    return matrix

def error_rate(cm):
    correct = np.trace(cm)
    total = np.sum(cm)
    return 1 - (correct / total)     
