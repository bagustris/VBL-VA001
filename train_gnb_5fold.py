# script to train VBL-VA001

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# load data hasil ekstraksi fitur fft
X = pd.read_csv("feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

# Setup arrays to store training and test accuracies
# SVM Machine Learning
# Setup arrays to store training and test accuracies
var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
test_accuracy = np.empty(len(var_gnb))

for i, k in enumerate(var_gnb):
    # Setup a Gaussian Naive Bayes Classifier
    clf_gnb = GaussianNB(var_smoothing=k)
    scores = cross_val_score(clf_gnb, X, y, cv=5)
    print(scores)
    # Compute accuracy on the test set
    test_accuracy[i] = np.mean(scores)

print(f"Max test acc: {np.max(test_accuracy)}")
max_var_gnb = np.argmax(test_accuracy)
print(f"Best var smoothing: {var_gnb[max_var_gnb]}")
