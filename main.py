from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth=1)
clf.fit(X, y)
print(clf.predict([[0, 0, 5, 1.5]]))