import numpy as np
# To make sure the target variable is ordinally encoded, use sklearn preprocessing
    # Can be commented out so algorithm functionality still works if y data is already in correct format
import warnings

class Node:
    """[Node class to build the decision tree with]

        [Predicted Class]:
            Target value with the most occurances
        [Feature Index]:
            Column index of the feature that the node splits on
        [Threshold]:
            Value of the feature that the node splits on
        [Left]:
            Node for under the threshold
        [Right]:
            Node for over the threshold
        [Leftbranch]:
            Is the node a left branch from its parent
        [Rightbranch]:
            Is the node a right branch from its parent
        [Depth]:
            Attribute used in testing to ensure that the print and fit methods were working correctly
                Still included in the grow_tree method but took out of print
    """
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
    # Tweaked a function for printing a binary tree here:
    # https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/#:~:text=1)%20Rightmost%20node%20is%20printed,fixed%20amount%20at%20every%20level.

    # Base case
    if root == None:
        return
    
    # Distance between levels
    space += count[0]

    # Go right first
    print_help(root.right, space)

    # Print current node after space
    print()
    for i in range(count[0], space):
        print(end = " ")
    if (root.left == None) and (root.right == None):
    # Print left or right leaf depending on if it's a left or right split
        if root.leftbranch == True:
            print(f"Left Leaf -> Predicted Class: {root.predicted_class}, Samples: {root.samples}")
        if root.rightbranch == True:
            print(f"Right Leaf -> Predicted Class: {root.predicted_class}, Samples: {root.samples}")
    else:
        msg = f"X{root.feature_index}, Threshold: {root.threshold}, Predicted Class: {root.predicted_class}, Samples: {root.samples}"
        print(msg)

    # Go left
    print_help(root.left, space)


class DecisionTreeClassifier:
    """[DecisionTreeClassifier class to fit a DTC and predict with it]

        [Max Depth]:
            Hyperparameter that controls how many branches deep the tree will stop growing at
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def print_tree(self):
        """[print function that prints out the attributes of the decision tree, if it's already fit]
        """
        node = self.tree
        # Check if root node is None
        if node == None:
            return "Tree not fit yet"
        # Recursively print each node in the tree using nodes str method
            # Base case = right and left == None
                # Return "X{feature_index}, Threshold: {threshold}, Predicted Class: {predicted class}"
        print_help(node, 0)

    def fit(self, X:np.array, y:np.array):
        """[Function to fit a decision tree classifier]

        Parameters
        ----------
        X : [NumPy Array]
            [rows of feature values]
        y : [NumPy Array]
            [target class value for each row]

        Attributes (of DecisionTreeClassifier, defined within fit function)
        ----------
            [Num Classes]
                Number of unique classes in the target
            [Num Features]
                Number of features in the data(X)
            [Tree]
                The decision tree
        """
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def find_split(self, X, y):
        """[Helper function to locate the ideal feature and threshold to split on. Called within grow_tree()]

        Parameters
        ----------
        X : [NumPy Array]
            [Training data, rows of features values]
        y : [NumPy Array]
            [Training target data, class values of each row]

        Returns
        -------
        ideal_col: [int]
                     [Column index of ideal feature to split on]
        ideal_threshold: [int]
                         [Ideal threshold value of the best feature to split on]
        """
        # Set default values of ideal column and threshold to None
        ideal_col = None
        ideal_threshold = None

        # Check to see if there are at least 1 observations
        num_observations = y.size
        if num_observations <= 1:
            return ideal_col, ideal_threshold
        
        # Reshape y back to shape (y.size,)
        y = y.reshape(num_observations,)
        
        # Find the count of each class for use in Gini calc later
            # To avoid warning message + future issues:
                #  locked the numpy and python versions
                #  suppress warning with default warnings library (still runs without it)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=FutureWarning)
            count_in_parent = [np.count_nonzero(y == c) for c in range(self.num_classes)]

        # Set default value of best_gini to the gini impurity value of the parent node
            # Good article on gini impurity here: https://victorzhou.com/blog/gini-impurity/
        best_gini = 1.0 - sum((n / num_observations) ** 2 for n in count_in_parent)

        # Create a temp version of Y in the right shape for concatenating
        temp_y = y.reshape(y.shape[0],1)
        # Loop through the columns in X
        # Sort X and y by values of X in each column
            # Allows us to more easily find the ideal threshold
                # Better time complexity -> https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
        for col in range(self.num_features):
            # Create a temp version of X[:,col] in the right shape for concatenating
            temp_X = X[:,col].reshape(num_observations,1)
            # Concatenate temp_X and temp_y
            all_data = np.concatenate((temp_X,temp_y), axis=1)
            # Sort the data using the colum with the feature/X value
            sorted_data = all_data[np.argsort(all_data[:,0])]
            # Split the data back into X and y, or threshold and classes values
            thresholds, obs_classes = np.array_split(sorted_data, 2, axis = 1)
            # Make sure observed classes are integers
            obs_classes = obs_classes.astype(int)
            
            # Keep track of how many of each class are going to each child node
                # Default is 0 of all classes to left and everything to right
            num_left = [0] * self.num_classes
            num_right = count_in_parent.copy()
            # Loop through all observations in the node to efficiently find
            # ideal threshold and class that minimizes Gini impurities of child nodes
            for i in range(1, num_observations):
                class_ = obs_classes[i - 1][0]
                num_left[class_] += 1
                num_right[class_] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (num_observations - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (num_observations - i) * gini_right) / num_observations
                # Go to the next i if i == i - 1
                    # Avoids making a split where two values are equal
                if thresholds[i][0] == thresholds[i - 1][0]:
                    continue
                # If gini is better than best_gini, re-assign ideals
                if gini < best_gini:
                    best_gini = gini
                    ideal_col = col
                    # Get the midpoint between i-th value and previous value
                    ideal_threshold = (thresholds[i][0] + thresholds[i - 1][0]) / 2
        return ideal_col, ideal_threshold

    def grow_tree(self, X, y, depth=0):
        """[Grow tree function to continue adding splits in the tree if depth < max_depth]

        Parameters
        ----------
        X : [np.Array]
            [X/feature values of current node]
        y : [np.Array]
            [y/target values of current node]
        depth : int, optional
            [depth of current node], by default 0

        Returns
        -------
        [Node]
            [Root Node of the DT]
        """
        # Get the population for each class of target in current node
            # Fix the same warning issue as before
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=FutureWarning)
            pop_per_class = [np.count_nonzero(y == i) for i in range(self.num_classes)]
        
        # predicted_class is the class w/ highest population
        predicted_class = np.argmax(pop_per_class)

        # Instantiate an empty Node
        node = Node(predicted_class=predicted_class,depth=depth)
        # Create a variable in the node class for the number of samples
        node.samples = y.size

        # If depth >= max_depth then leave it empty and return it
            # Else
                # Grow the tree recursively
                    # Find the new best split
                    # Use the new subset of X and y to grow the tree
                    # Make left/right branch True
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

    def predict(self, X_test):
        """[Function to predict using new X input(s)]

        Parameters
        ----------
        X_test : [np.Array (shape of (number of rows/observations,X_train.shape[1])) or Python list/list of lists]
            [X/feature test data]

        Returns
        -------
        [np.Array of shape (observations, )]
            [Predicted target class(es) of the test data]
        """
        node = self.tree
        predictions = []
        for obs in X_test:
            # Have to reassign node to root node or else will make
            # all the same predictions!!!
            node = self.tree
            while node.left:
                if obs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predicted_class)
        return np.array(predictions)