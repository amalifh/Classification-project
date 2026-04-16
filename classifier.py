import numpy as np
import pandas as pd
def score_function(weight, bias, data):
    w_t = np.transpose(weight)
    w_tx = np.matmul(w_t, data)
    return w_tx + bias
   
def check_label_data(file):
    df = pd.read_csv(file)
    df['iris_type'] = df['iris_type'].str.lower().map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
    })
    return df

def MSE(prediction, target):
    return (prediction-target)*(prediction-target)

def update_weights(prediction, target):
    

#three separate binary classifiers
def iris_classifier(file):
    feature = np.zeros(4)
    df = check_label_data(file)
    print(df)
    w1 = np.zeros(4)
    w2 = np.zeros(4)
    w3 = np.zeros(4)

    b1 = 0
    b2 = 0
    b3 = 0

    alpha = 0.01

    for _, row in df.iterrows():
        feature = row[:4]
        target = row.iloc[4]
        score1 = score_function(w1, b1, feature)
        score2 = score_function(w2, b2, feature)
        score3 = score_function(w3, b3, feature)
        scores = [score1, score2, score3]
        predicted = np.argmax(scores)
        
        #if wrong, increase weight of true and decrease weight of predicted
        if predicted != target:
            error = predicted - target
            MSE = error * error
            gradient_w = error * feature

            if predicted == 0:
                w1 = w1
    print(f'predicted: {predicted} and target {target}')

iris_classifier('class_2_training')
