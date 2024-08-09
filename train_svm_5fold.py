# Cross validation 5 folds SVM evaluation
# Compare this snippet from train_svm.py:

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# load data hasil ekstraksi fitur fft
X = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

# Setup arrays to store training and test accuracies
c_svm = np.arange(1, 100)
test_accuracy = np.empty(len(c_svm))

# finding best c for five folds
for i, k in enumerate(c_svm):
    # Setup a knn classifier with c_svm
    clf_svm = SVC(C=k)
    # Do 5-cv to the model
    scores = cross_val_score(clf_svm, X, y, cv=5)
    print(scores)
    # Compute accuracy on the test set
    test_accuracy[i] = np.mean(scores)

# print max test accuracy (average of 5 folds)
print(f"Max test acc: {np.max(test_accuracy)}")
print(f"Best C: {np.argmax(test_accuracy)+1}")

