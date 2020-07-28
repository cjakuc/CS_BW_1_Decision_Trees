from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth=1)
clf.fit(X, y)
temp = np.array([[0, 0, 5, 1.5]])
print(clf.predict(temp))