from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth=1)
y = ["one" if val == 1 or val == 2 else "zero" for val in y]
y = np.array(y)
print("------")
print(str(y.dtype))
print("------")

# if "int" not in str(y.dtype):
#     # Reshape y array so it works w/ ordinal encoder
#     y = y.reshape(-1, 1)
#     encoder = OrdinalEncoder()
#     y = encoder.fit_transform(y)
# y = y.astype(int)

clf.fit(X, y)
# print(y)
# print("------")
# print(y.dtype)
# print("------")
temp = np.array([[0, 0, 5, 1.5]])
print(clf.predict(temp))

