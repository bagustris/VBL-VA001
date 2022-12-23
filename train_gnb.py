# script to train VBL-VA001

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# load data hasil ekstraksi fitur fft
x = pd.read_csv("feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)


print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# kNN Machine Learning
# import KNeighborsClassifier

# Setup arrays to store training and test accuracies
# SVM Machine Learning
# Setup arrays to store training and test accuracies
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

# print max acccuracy
print(f"Max test acc: {np.max(test_accuracy)}")

# Generate plot
# plt.title('Varying var_smoothing in GNB')
plt.plot(var_gnb, test_accuracy, label='Testing Accuracy')
plt.plot(var_gnb, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('var_smoothing')
plt.ylabel('Accuracy')
# np.savetxt('gnb_var.txt', test_accuracy)
plt.savefig('acc_GNB.pdf')
# plt.show()
