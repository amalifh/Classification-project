import utils
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture

N_classes = 12
cols = [
    "filename","duration","f0",
    "F1","F2","F3","F4",
    "F1_20","F2_20","F3_20",
    "F1_50","F2_50","F3_50",
    "F1_80","F2_80","F3_80"
]

classes = ['ae', 'ah', 'aw', 'eh', 'er',
           'ei', 'ih', 'iy', 'oa', 'oo',
           'uh', 'uw']


def main():
    class_data = utils.clean_df("Vowels/vowdata_nohead.dat", cols)
    means = {}
    covariances = {}
    for vow in classes:
        mu, cov = utils.mean_covariance(class_data, vow)
        means[vow] = mu
        covariances[vow] = cov
    mean_matrix = pd.DataFrame.from_dict(
    means,
    orient="index",
    columns=["F1","F2","F3"]
)


    for cov in covariances:
        cov_matrix = pd.DataFrame(
        covariances[cov],
        index = ['F1', 'F2', 'F3'],
        columns=["F1","F2","F3"]
    )
#        print(f'Covariance matrix for class: {cov}')
#        print(cov_matrix)

#------Confusion matrix------
    cm = np.zeros((N_classes, N_classes))
    class_to_idx = {
        v:i for i,v in enumerate(classes)
    }
    for target in classes:
        test = class_data[target].iloc[:70]
        X_test = test[["F1","F2","F3"]].to_numpy()

        for x in X_test:
            predicted = utils.classifier(x, means, covariances)

            i = class_to_idx[target]
            j = class_to_idx[predicted]

            cm[i,j] += 1
            #make again for diag cov

    error = utils.error_rate(cm)
    print(f'Error rate: {error}')

    diag_covariances ={}
    for v in covariances:
        diag_covariances[v] = np.diag(
            np.diag(covariances[v])
        )

    cm_diag = np.zeros((N_classes,N_classes))

    for target in classes:

        test = class_data[target].iloc[70:]
        X_test = test[["F1","F2","F3"]].to_numpy()

        for x in X_test:

            predicted = utils.classifier(
                x,
                means,
                diag_covariances
            )

            i = class_to_idx[target]
            j = class_to_idx[predicted]

            cm_diag[i,j] += 1

    error_diag = 1 - np.trace(cm_diag)/np.sum(cm_diag)
    print(f'Error rate digonal covariance matrix: {error_diag}')

#---------------------TASK 2------------------

    gmms_2 = utils.train_gmms(class_data, classes, M=2)
    cm_2 = np.zeros((N_classes, N_classes))

    gmms_3 = utils.train_gmms(class_data, classes, M=3)
    cm_3 = np.zeros((N_classes, N_classes))
    for target in classes:
        test = class_data[target].iloc[70:]
        X_test = test[["F1", "F2", "F3"]].to_numpy()

        for x in X_test:
            predicted_2 = utils.gmm_classifier(x, gmms_2)
            predicted_3 = utils.gmm_classifier(x, gmms_3)

            i = class_to_idx[target]
            j = class_to_idx[predicted_2]
            k = class_to_idx[predicted_3]

            cm_2[i,j] += 1
            cm_3[i,k] += 1
    
    print(utils.error_rate(cm_2))
    print(utils.error_rate(cm_3))
    disp_training = ConfusionMatrixDisplay(confusion_matrix=cm_3, display_labels=classes)
    disp_training.plot(cmap=plt.cm.Reds)
    plt.title('Confusion Matrix')
    plt.savefig("Vowels/figures/CM_3", dpi = 500)


if __name__ == "__main__":
    main()