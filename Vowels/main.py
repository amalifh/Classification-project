import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


N_CLASSES = 12

COLUMNS = [
    "filename", "duration", "f0",
    "F1", "F2", "F3", "F4",
    "F1_20", "F2_20", "F3_20",
    "F1_50", "F2_50", "F3_50",
    "F1_80", "F2_80", "F3_80"
]

CLASSES = [
    "ae", "ah", "aw", "eh", "er", "ei",
    "ih", "iy", "oa", "oo", "uh", "uw"
]


def print_means_and_covariances(means, covariances):
    mean_matrix = pd.DataFrame.from_dict(
        means,
        orient="index",
        columns=["F1", "F2", "F3"]
    )

    print("\nSample means:")
    print(mean_matrix.round(2))

    for vow in CLASSES:
        cov_matrix = pd.DataFrame(
            covariances[vow],
            index=["F1", "F2", "F3"],
            columns=["F1", "F2", "F3"]
        )

        print(f"\nCovariance matrix for class {vow}:")
        print(cov_matrix.round(2))


def plot_confusion_matrix(cm, title, filename):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    disp.plot(cmap=plt.cm.Reds)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.show()


def main():
    class_data = utils.clean_df(
        "Vowels/vowdata_nohead.dat",
        COLUMNS
    )

    train_data, test_data = utils.split_data(
        class_data,
        CLASSES,
        n_train=70,
        random_state=0
    )

#-----------------TASK ONE-----------------
    means_full, cov_full = utils.train_single_gaussian(
        train_data,
        CLASSES,
        diagonal=False
    )

    cm_full = utils.test_single_gaussian(
        test_data,
        CLASSES,
        means_full,
        cov_full
    )

    means_diag, cov_diag = utils.train_single_gaussian(
        train_data,
        CLASSES,
        diagonal=True
    )

    cm_diag = utils.test_single_gaussian(
        test_data,
        CLASSES,
        means_diag,
        cov_diag
    )

#-----------------TASK TWO-----------------
    gmms_2 = utils.train_gmms(
        train_data,
        CLASSES,
        M=2
    )

    cm_gmm_2 = utils.test_gmms(
        test_data,
        CLASSES,
        gmms_2
    )

    gmms_3 = utils.train_gmms(
        train_data,
        CLASSES,
        M=3
    )

    cm_gmm_3 = utils.test_gmms(
        test_data,
        CLASSES,
        gmms_3
    )

    print_means_and_covariances(means_full, cov_full)

    print(f"\nError rate full covariance: {utils.error_rate(cm_full) * 100:.2f}%")
    print(f"Error rate diagonal covariance: {utils.error_rate(cm_diag) * 100:.2f}%")
    print(f"Error rate 2 mixtures: {utils.error_rate(cm_gmm_2) * 100:.2f}%")
    print(f"Error rate 3 mixtures: {utils.error_rate(cm_gmm_3) * 100:.2f}%")

    plot_confusion_matrix(
        cm_full,
        f"CM full covariance\nError rate {utils.error_rate(cm_full) * 100:.2f}%",
        "Vowels/figures/CM_full_covariance.png"
    )

    plot_confusion_matrix(
        cm_diag,
        f"CM diagonal covariance\nError rate {utils.error_rate(cm_diag) * 100:.2f}%",
        "Vowels/figures/CM_diagonal_covariance.png"
    )

    plot_confusion_matrix(
        cm_gmm_2,
        f"CM GMM 2 mixtures\nError rate {utils.error_rate(cm_gmm_2) * 100:.2f}%",
        "Vowels/figures/CM_gmm_2.png"
    )

    plot_confusion_matrix(
        cm_gmm_3,
        f"CM GMM 3 mixtures\nError rate {utils.error_rate(cm_gmm_3) * 100:.2f}%",
        "Vowels/figures/CM_gmm_3.png"
    )


if __name__ == "__main__":
    main()