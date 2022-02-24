# Some of this code is taken from Ch. 3 of Hands-on Machine Learning by Geron.
import sys
import pandas as pd
assert sys.version_info >= (3, 5)


# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
print(len(X), len(y))

# To plot pretty figures



mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

"""
This function runs multiple experiments on multiple classifiers with
the training data amounts determined by the list num_training_examples.
The results are then plotted.
"""
def run_experiments(X : list, y : list, classifiers : list, regularization):
    num_training_examples = 60000
    C_values = [1 / x for x in regularization]
    X_train, X_test, y_train, y_test = X[:1000], X[60000:], y[:1000], y[60000:]
    print(num_training_examples)
    all_accuracies = pd.DataFrame()
    #classifier_name = type(clf).__name__
    clf_accuracies_l1 = []
    clf_accuracies_l2 = []
    for C in C_values:
        #l1 regularization
        sys.stdout.write("Training LR with l1 regularization on lambda =  " + str(1/C) + "\n")
        clf_l1 = LogisticRegression(n_jobs=-1, penalty='elasticnet', solver='saga', tol=0.1, C = C, l1_ratio=0.5)
        clf_l1.fit(X_train, y_train)
        y_pred = clf_l1.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        sys.stdout.write("L1 Accuracy " + str(accuracy) + "\n")
        clf_accuracies_l1.append(accuracy) # for this classifier
        #plot_confusion_matrix(confusion_matrix(y_test, y_pred), clf)
        #plt.show()
        print(clf_accuracies_l1)

        #l2 regularization
        sys.stdout.write("Training LR with l2 regularization on lambda =  " + str(1 / C) + "\n")
        clf_l2 = LogisticRegression(n_jobs=-1, penalty='l2', solver='saga', tol=0.1, C=C)
        clf_l2.fit(X_train, y_train)
        y_pred = clf_l2.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        sys.stdout.write("L2 Accuracy " + str(accuracy) + "\n")
        clf_accuracies_l2.append(accuracy)  # for this classifier
        # plot_confusion_matrix(confusion_matrix(y_test, y_pred), clf)
        # plt.show()
        print(clf_accuracies_l2)
    l1_accuracy = pd.Series(clf_accuracies_l1, index=regularization, name="l1 regularization")
    l2_accuracy = pd.Series(clf_accuracies_l2, index=regularization, name="l2 regularization")
    plot_accuracy_l1 = pd.DataFrame(l1_accuracy)
    plot_accuracy_l2 = pd.DataFrame(l2_accuracy)
    all_accuracies = all_accuracies.append(l1_accuracy)
    all_accuracies = all_accuracies.append(l2_accuracy)# for all classifiers

    plot_accuracy_l1.plot(style = '.-')
    plot_accuracy_l2.plot(style='.-')
    #plot = sns.scatterplot(data=all_accuracies.transpose())
    print(all_accuracies.head())
    all_accuracies.transpose().plot(style='.-')
    plt.show()
   # plot_confusion_matrix(confusion_matrix(y_test, y_pred))

def plot_confusion_matrix(matrix, classifier):
    """If you prefer color and a colorbar"""
    classifiers = {"LogisticRegression(n_jobs=-1, penalty='l1', solver='saga', tol=0.1)": "Logistic Regression with L1 Regularization",
                   "LogisticRegression(n_jobs=-1, penalty='l2, solver='saga', tol=0.1)": "Logistic Regression with L2 Regularization",
                   "LogisticRegression(n_jobs=-1, penalty='elasicnet', solver='saga', tol=0.1)": "Logistic Regression with elastic net"}
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title(classifiers[str(classifier)])
    fig.colorbar(cax)


def save_figs(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


if __name__ == "__main__":
    #scikit learn C = 1/lambda
    regularization = [x / 1000 for x in range(1, 100, 1)]
    print(regularization)
    lr_l1 = LogisticRegression(solver="saga", tol=0.1, penalty="l1", n_jobs=-1)
    lr_l2 = LogisticRegression(solver="saga", tol=0.1, penalty="l2", n_jobs=-1)
    lr_elastic_net = LogisticRegression(solver="saga", tol=0.1, penalty="elasticnet", n_jobs=-1)
    classifiers = [lr_l1, lr_l2]
    
    run_experiments(X, y, classifiers, regularization)
    plt.show()
    
    print("Done.")



#ovr_clf.predict([some_digit])
