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


def mean_covariance(data):
    X = data[["F1","F2","F3"]].to_numpy()
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)

    return mu, Sigma


def classifier(x, means, covs):
    best_class = None
    best_score = -np.inf

    for vow in means:
        mu = means[vow]
        Sigma = covs[vow] + 1e-6 * np.eye(3)

        diff = x - mu

        score = (
            -0.5 * np.log(np.linalg.det(Sigma))
            -0.5 * diff.T @ np.linalg.inv(Sigma) @ diff
        )

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
        X_train = data[["F1","F2","F3"]].to_numpy()

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

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


FEATURES = ["F1", "F2", "F3"]


def clean_df(file, cols):
    df = pd.read_csv(file, sep=r"\s+", names=cols)

    df["vowel"] = df["filename"].str[-2:]
    df = df[["vowel", "F1", "F2", "F3"]]

    df = df[
        (df["F1"] != 0) &
        (df["F2"] != 0) &
        (df["F3"] != 0)
    ]

    data = {}
    for vow in df["vowel"].unique():
        data[vow] = df[df["vowel"] == vow].reset_index(drop=True)

    return data


def split_data(class_data, classes, n_train=70, random_state=0):
    train_data = {}
    test_data = {}

    for vow in classes:
        df = class_data[vow].sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_data[vow] = df.iloc[:n_train]
        test_data[vow] = df.iloc[n_train:]

    return train_data, test_data


def mean_covariance(df):
    X = df[FEATURES].to_numpy()
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    return mu, cov


def train_single_gaussian(train_data, classes, diagonal=False):
    means = {}
    covariances = {}

    for vow in classes:
        mu, cov = mean_covariance(train_data[vow])

        if diagonal:
            cov = np.diag(np.diag(cov))

        means[vow] = mu
        covariances[vow] = cov

    return means, covariances


def classifier(x, means, covariances):
    best_class = None
    best_score = -np.inf

    for vow in means:
        mu = means[vow]
        cov = covariances[vow] + 1e-6 * np.eye(3)

        diff = x - mu

        score = (
            -0.5 * np.log(np.linalg.det(cov))
            -0.5 * diff.T @ np.linalg.inv(cov) @ diff
        )

        if score > best_score:
            best_score = score
            best_class = vow

    return best_class


def test_single_gaussian(test_data, classes, means, covariances):
    cm = np.zeros((len(classes), len(classes)))
    class_to_idx = {vow: i for i, vow in enumerate(classes)}

    for target in classes:
        X_test = test_data[target][FEATURES].to_numpy()

        for x in X_test:
            predicted = classifier(x, means, covariances)

            i = class_to_idx[target]
            j = class_to_idx[predicted]

            cm[i, j] += 1

    return cm


def train_gmms(train_data, classes, M):
    gmms = {}

    for vow in classes:
        X_train = train_data[vow][FEATURES].to_numpy()

        gmm = GaussianMixture(
            n_components=M,
            covariance_type="diag",
            random_state=0,
            reg_covar=1e-6,
            n_init=10,
            max_iter=500
        )

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


def test_gmms(test_data, classes, gmms):
    cm = np.zeros((len(classes), len(classes)))
    class_to_idx = {vow: i for i, vow in enumerate(classes)}

    for target in classes:
        X_test = test_data[target][FEATURES].to_numpy()

        for x in X_test:
            predicted = gmm_classifier(x, gmms)

            i = class_to_idx[target]
            j = class_to_idx[predicted]

            cm[i, j] += 1

    return cm


def error_rate(cm):
    return 1 - np.trace(cm) / np.sum(cm)