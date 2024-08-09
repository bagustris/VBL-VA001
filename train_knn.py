# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# load data hasil ekstraksi fitur fft
x = pd.read_csv('feature_VBL-VA001.csv', header=None)

# load label
y = pd.read_csv('label_VBL-VA001.csv', header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True)


print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# kNN Machine Learning
# import KNeighborsClassifier
# Setup arrays to store training and test accuracies
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

# print max acccuracy
print(f"Max test acc: {np.max(test_accuracy)}")

# Generate plot
# plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
# np.savetxt('knn_n.txt', test_accuracy)
# plt.savefig('acc_knn.pdf')

# print optimal k and max test accuracy
print(f"Optimal k: {np.argmax(test_accuracy)}")
print(f"Max test accuracy: {max(test_accuracy)}")
