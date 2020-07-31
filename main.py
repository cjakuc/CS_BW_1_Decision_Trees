from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Test w/ Iris dataset using my class
dataset = load_iris()
X, y = dataset.data, dataset.target
clf_iris = DecisionTreeClassifier(max_depth = 5)
# Test to make target class strings instead of integers
y = ["one" if val == 1 or val == 2 else "zero" for val in y]
y = np.array(y)
# Need to ordinally encode strings to integers
if "int" not in str(y.dtype):
    # Reshape y array so it works w/ ordinal encoder
    y = y.reshape(-1, 1)
    encoder = OrdinalEncoder()
    y = encoder.fit_transform(y)
y = y.astype(int)
y = y.reshape(y.size,)

clf_iris.fit(X, y)
temp = np.array([[3, 2, 1, .5]])
print("My Iris DT:")
clf_iris.print_tree()
print("------------------------------------------------------")
print(f"My Iris prediction for {temp}:\n", clf_iris.predict(temp))
print("------------------------------------------------------")
# Test w/ Iris dataset using sklearn
skl_clf_iris = DTC(splitter="best",random_state=42, max_depth=5)
skl_clf_iris.fit(X,y)
skl_preds_iris = skl_clf_iris.predict(temp)
print(f"SKLearn Iris prediction for {temp}:\n",skl_preds_iris)
print("------------------------------------------------------")
# Save an image of the SKL Iris tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(skl_clf_iris, class_names=True);
fig.savefig('SKL_Iris_DT.png')

# Test w/ Wine dataset using my class
dataset = load_wine()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)
clf_wine = DecisionTreeClassifier(max_depth = 5)
clf_wine.fit(X_train, y_train)
preds = clf_wine.predict(X_test)
print("My Wine DT:")
clf_wine.print_tree()
print("------------------------------------------------------")
print("My Wine predictions:\n",preds)
print("------------------------------------------------------")
# Get the accuracy
correct_count = len(set(preds) & set(y_test))
wine_accuracy = correct_count / len(y_test)
print("My Wine accuracy:\n", wine_accuracy)
print("------------------------------------------------------")
# Test w/ Wine dataset using SKLearn
skl_clf_wine = DTC(splitter="best",random_state=42, max_depth=4)
skl_clf_wine.fit(X_train,y_train)
skl_preds = skl_clf_wine.predict(X_test)
print("SKLearn Wine predictions:\n",skl_preds)
print("------------------------------------------------------")
# Get the accuracy
skl_correct_count = len(set(skl_preds) & set(y_test))
skl_wine_accuracy = skl_correct_count / len(y_test)
print("SKL Wine accuracy:\n", skl_wine_accuracy)
print("------------------------------------------------------")
# Save an image of the SKL Wine tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(skl_clf_wine, class_names=True);
fig.savefig('SKL_Wine_DT.png')
 