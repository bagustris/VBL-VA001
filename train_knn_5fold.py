# script to train VBL-VA001, 5-cv knn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# load data hasil ekstraksi fitur fft
X = pd.read_csv('feature_VBL-VA001.csv', header=None)

# load label
y = pd.read_csv('label_VBL-VA001.csv', header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

# import KNeighborsClassifier
# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 100)
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf_knn, X, y, cv=5)
    print(scores)
    # Compute average accuracy on the test set
    test_accuracy[i] = np.mean(scores)

# print max test accuracy (average of 5 folds)
print(f"Max test acc: {np.max(test_accuracy)}")
print(f"Best neighbors: {np.argmax(test_accuracy)+1}")
