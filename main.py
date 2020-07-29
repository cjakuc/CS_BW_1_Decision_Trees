from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
import matplotlib.pyplot as plt


# dataset = load_iris()
# X, y = dataset.data, dataset.target
# clf_iris = DecisionTreeClassifier(max_depth = 5)
# y = ["one" if val == 1 or val == 2 else "zero" for val in y]
# y = np.array(y)
# # Make sure that the target, y, values are ordinally encoded integers
#     # Reshape y array so it works w/ ordinal encoder
# y = y.reshape(-1, 1)
# encoder = OrdinalEncoder()
# y = encoder.fit_transform(y)
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
# clf_iris.fit(X, y)
# temp = np.array([[0, 0, 5, 1.5]])
# print(clf.predict(temp))
# print(str(clf_iris))

print("------")


dataset = load_wine()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)
clf_wine = DecisionTreeClassifier(max_depth = 5)
clf_wine.fit(X_train, y_train)
preds = clf_wine._predict(X_test, multi_obs=True)
print(str(clf_wine))
print("------")
print(preds)
print("------")
# print(y_test)
# print("------")
# print(preds.shape)
# print(min(X_test[:,5]))

skl_clf = DTC(splitter="best",random_state=42, max_depth=4)
skl_clf.fit(X_train,y_train)
skl_preds = skl_clf.predict(X_test)
print(skl_preds)
print("------")
print(X_test[1])
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(skl_clf, class_names=True);
fig.savefig('imagename.png')
# unique_elements, counts_elements = np.unique(y_test, return_counts=True)
# print("Frequency of unique values of the said array:")
# print(np.asarray((unique_elements, counts_elements)))