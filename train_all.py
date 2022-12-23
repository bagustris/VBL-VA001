# script to train VBL-VA001

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# load data hasil ekstraksi fitur fft
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# Setup arrays to store training and test accuracies
c_svm = np.arange(1, 100)
train_accuracy = np.empty(len(c_svm))
test_accuracy = np.empty(len(c_svm))

for i, k in enumerate(c_svm):
    # Setup a knn classifier with c_svm
    svm = SVC(C=k)
    # Fit the model
    svm.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = svm.score(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = svm.score(X_test, y_test)

# KNN
neighbors = np.arange(1, 100)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
    knn.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
train_accuracy = np.empty(len(var_gnb))
test_accuracy = np.empty(len(var_gnb))

for i, k in enumerate(var_gnb):
    # Setup a Gaussian Naive Bayes Classifier
    model = GaussianNB(var_smoothing=k)
    gnb = model.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = gnb.score(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = gnb.score(X_test, y_test)

# plot all three
plt.subplot(1, 3, 1)
plt.plot(c_svm, test_accuracy, label='Testing Accuracy')
plt.plot(c_svm, train_accuracy, label='Training accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 2)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(var_gnb, test_accuracy, label='Testing Accuracy')
plt.plot(var_gnb, train_accuracy, label='Training accuracy')
plt.xlabel('var_smoothing')
plt.ylabel('Accuracy')

# plt.savefig('plot_all.pdf')
plt.show()