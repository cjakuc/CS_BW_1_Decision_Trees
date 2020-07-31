import numpy as np
import warnings

class Node:
    def __init__(self, predicted_class, depth=None):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.leftbranch = False
        self.rightbranch = False
        self.depth = depth
    
def print_help(root, space, count=[10]):
    if root == None:
        return
    
    space += count[0]

    print_help(root.right, space)

    print()
    for i in range(count[0], space):
        print(end = " ")
    if (root.left == None) and (root.right == None):
        if root.leftbranch == True:
            print(f"Left Leaf -> Predicted Class: {root.predicted_class}, Samples: {root.samples}")
        if root.rightbranch == True:
            print(f"Right Leaf -> Predicted Class: {root.predicted_class}, Samples: {root.samples}")
    else:
        msg = f"X{root.feature_index}, Threshold: {root.threshold}, Predicted Class: {root.predicted_class}, Samples: {root.samples}"
        print(msg)

    print_help(root.left, space)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def print_tree(self):
        node = self.tree
        if node == None:
            return "Tree not fit yet"
        print_help(node, 0)
        return ""

    def fit(self, X:np.array, y:np.array):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def find_split(self, X, y):
        num_observations = y.size
        if num_observations <= 1:
            return None, None
        
        y = y.reshape(num_observations,)
        
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=FutureWarning)
            count_in_parent = [np.count_nonzero(y == c) for c in range(self.num_classes)]

        best_gini = 1.0 - sum((n / num_observations) ** 2 for n in count_in_parent)

        ideal_col = None
        ideal_threshold = None
        temp_y = y.reshape(y.shape[0],1)

        for col in range(self.num_features):
            temp_X = X[:,col].reshape(num_observations,1)
            all_data = np.concatenate((temp_X,temp_y), axis=1)
            sorted_data = all_data[np.argsort(all_data[:,0])]
            thresholds, obs_classes = np.array_split(sorted_data, 2, axis = 1)
            obs_classes = obs_classes.astype(int)
            

            num_left = [0] * self.num_classes
            num_right = count_in_parent.copy()

            for i in range(1, num_observations):
                class_ = obs_classes[i - 1][0]
                num_left[class_] += 1
                num_right[class_] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (num_observations - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (num_observations - i) * gini_right) / num_observations

                if thresholds[i][0] == thresholds[i - 1][0]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    ideal_col = col
                    ideal_threshold = (thresholds[i][0] + thresholds[i - 1][0]) / 2
        return ideal_col, ideal_threshold

    def grow_tree(self, X, y, depth=0):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=FutureWarning)
            pop_per_class = [np.count_nonzero(y == i) for i in range(self.num_classes)]
        
        predicted_class = np.argmax(pop_per_class)

        node = Node(predicted_class=predicted_class,depth=depth)
        node.samples = y.size

        if depth < self.max_depth:
            col, threshold = self.find_split(X, y)
            if col and threshold:
                indices_left = X[:, col] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                indices_right = X[:, col] >= threshold
                X_right, y_right = X[indices_right], y[indices_right]
                node.feature_index = col
                node.threshold = threshold
                node.left = self.grow_tree(X_left, y_left, depth+1)
                node.left.leftbranch = True
                node.right = self.grow_tree(X_right, y_right, depth+1)
                node.right.rightbranch = True
        return node

    def _predict(self, X_test):

        node = self.tree
        predictions = []
        for obs in X_test:
            node = self.tree
            while node.left:
                if obs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predicted_class)
        return np.array(predictions)