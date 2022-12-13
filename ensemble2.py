import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(max_iter=1000, random_state=123, solver="liblinear")
clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
clf3 = GaussianNB()

#X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
#y = np.array([1, 1, 2, 2])

# load data hasil ekstraksi fitur fft
X = pd.read_csv("feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

print(f"Shape of Train Data: ​{X_train.shape}")
print(f"Shape of Test Data: {X_test.shape}")

eclf = VotingClassifier(
    estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
    voting="soft",
    weights=[1, 1],
)

# predict class probabilities for all classifiers
# probas = [c.fit(X_train, y_train).predict_proba(X_test) for c in (clf1, clf2, clf3, eclf)]
out_vot = eclf.fit(X_train, y_train)
probas = out_vot.predict_proba(X_test)
train_score = eclf.score(X_train, y_train)
test_score = eclf.score(X_test, y_test)

# plotting
from sklearn.metrics import plot_confusion_matrix

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10)
# plot_confusion_matrix(eclf, X_test, y_test, normalize="true")
# plt.show()
print(f"Test accuracy: {test_score}")
print(f"Test accuracy: {test_score}")

# clf1 + clf3 = 0.995
# clf1 + clf2 = 1
# clf2 + clf3 = 0.998
