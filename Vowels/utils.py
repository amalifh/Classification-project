import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def clean_df(file, cols):
    df = pd.read_csv(file, sep=r"\s+", names=cols
)
    df["vowel"] = df["filename"].str[-2:]
    df = df[["vowel","F1","F2","F3"]]
    df = df[(df["F1"] != 0) &
        (df["F2"] != 0) &
        (df["F3"] != 0)]
    data = {}
    for vow in df["vowel"].unique():
        data[vow] = df[df["vowel"] == vow]

    return data


def mean_covariance(classes, vow_class):
    vow_train = classes[vow_class].iloc[:70]
    X = vow_train[["F1","F2","F3"]].to_numpy()
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)

    return mu, Sigma


def classifier(x, means, covs):
    best_class = None
    best_score = -np.inf

    for vow in means:
        mu = means[vow]
        Sigma = covs[vow]

        diff = x - mu
        score = (-0.5*np.log(np.linalg.det(Sigma))-0.5*diff.T @ np.linalg.inv(Sigma) @ diff)
        
        if score > best_score:
            best_score = score
            best_class = vow

    return best_class

def error_rate(cm):
    correct = np.trace(cm)
    total = np.sum(cm)
    return 1 - (correct / total)

def train_gmms(data, classes, M):
    gmms = {}

    for vow in classes:
        train = data[vow].iloc[:70]
        X_train = train[["F1","F2","F3"]].to_numpy()

        gmm = GaussianMixture(n_components= M, covariance_type='full', random_state=42)
        gmm.fit(X_train)
        gmms[vow] = gmm

    return gmms

def gmm_classifier(x, gmms):
    best_class = None
    best_score = -np.inf

    x = x.reshape(1, -1)

    for vow in gmms:
        score = gmms[vow].score_samples(x)[0]

        if score > best_score:
            best_score = score
            best_class = vow

    return best_class


def cm_gmm(data, classes, gmms):
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))

    class_to_idx = {vow: i for i, vow in enumerate(classes)}

    for vow in classes:
        test = data[vow].iloc[70:]
        X_test = test[["F1", "F2", "F3"]].to_numpy()

        for x in X_test:
            predicted = gmm_classifier(x, gmms)

            i = class_to_idx[vow]
            j = class_to_idx[predicted]

            cm[i, j] += 1
    return cm