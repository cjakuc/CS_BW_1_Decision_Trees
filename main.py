from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth = 5)
y = ["one" if val == 1 or val == 2 else "zero" for val in y]
y = np.array(y)
# print("------")
# print(str(y.dtype))
# print("------")

# if "int" not in str(y.dtype):
#     # Reshape y array so it works w/ ordinal encoder
#     y = y.reshape(-1, 1)
#     encoder = OrdinalEncoder()
#     y = encoder.fit_transform(y)
# y = y.astype(int)
# y = y.reshape(y.size,)
# y = y.reshape(150,1)
# # print(y)
# print("------")
# temp_X = X[:,1].reshape(X.shape[0],1)
# all_data = np.concatenate((temp_X,y),axis=1)
# sorted_data = all_data[np.argsort(all_data[:,0])]
# threshold, obs_classes = np.array_split(sorted_data, 2, axis = 1)
# # all_data = np.concatenate((X,y), axis=1)
# # print(X[:,0].reshape(X.shape[0],1))
# # print("------")
# print(threshold[0][0])
# print(obs_classes[0][0])
# print("------")
# print(y.dtype)
# print("------")
clf.fit(X, y)
# temp = np.array([[0, 0, 5, 1.5]])
# print(clf.predict(temp))
print(str(clf))

