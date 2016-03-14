import subprocess

import numpy as np
from pandas import DataFrame, read_csv, Series
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from colors import bcolors
from constructors.treeconstructor import TreeConstructor
from decisiontree import DecisionTree


class CARTconstructor(TreeConstructor):
    def split_criterion(self, node):
        raise NotImplementedError(
            "This method is not implemented, because we use the optimised sklearn pruning algorithm")

    def __init__(self):
        pass

    def cross_validation(self, data, k):
        return KFold(len(data.index), n_folds=k, shuffle=True)

    def construct_tree(self, training_feature_vectors, labels):
        """
        This method constructs an sklearn decision tree and trains it with the training data
        The sklearn decision tree classifier is stored in the CARTconstructor.dt

        :param training_feature_vectors: the feature vectors of the training samples
        :param labels: the labels of the training samples
        :return: void
        """
        self.features = list(training_feature_vectors.columns)
        # print"* features:", self.features

        self.y = labels['cat']
        self.X = training_feature_vectors[self.features]

        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X, self.y)

    def calculate_error_rate(self, tree, testing_feature_vectors, labels):
        return 1-tree.dt.score(testing_feature_vectors, labels)

    def post_prune(self, tree, testing_feature_vectors, labels, significance=0.125):
        pass

    def visualize_tree(tree, feature_names, labelnames, filename):
        """Create tree png using graphviz.

        Args
        ----
        tree -- scikit-learn DecsisionTree.
        feature_names -- list of feature names.
        """
        labels = Series(labelnames.values.ravel()).unique()
        labels.sort()
        labels = map(str, labels)
        # labels = labelnames.unique()
        # print labels
        with open(filename + ".dot", 'w') as f:
            export_graphviz(tree.dt, out_file=f,
                            feature_names=feature_names, class_names=labels)

        command = ["dot", "-Tpdf", filename + ".dot", "-o", filename + ".pdf"]
        try:
            subprocess.check_call(command)
        except:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")

    def convertToTree(self, verbose=False):
        #       # Using those arrays, we can parse the tree structure:


        # label = naam feature waarop je splitst
        # value = is de value van de feature waarop je splitst
        # ownDecisionTree.


        n_nodes = self.dt.tree_.node_count
        children_left = self.dt.tree_.children_left
        children_right = self.dt.tree_.children_right
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        classes = self.dt.classes_
        print classes

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes)
        decision_trees = [None] * n_nodes
        for i in range(n_nodes):
            decision_trees[i] = DecisionTree()
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        if verbose:
            print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)

        for i in range(n_nodes):

            if children_left[i] > 0:
                decision_trees[i].left = decision_trees[children_left[i]]

            if children_right[i] > 0:
                decision_trees[i].right = decision_trees[children_right[i]]

            if is_leaves[i]:
                # decision_trees[i].label = self.dt.classes_[self.dt.tree_.value[i][0][1]]
                decision_trees[i].label = self.dt.classes_[np.argmax(self.dt.tree_.value[i][0])]
                decision_trees[i].value = None
                if verbose:
                    print(bcolors.OKBLUE + "%snode=%s leaf node." % (node_depth[i] * "\t", i)) + bcolors.ENDC
            else:
                decision_trees[i].label = self.features[feature[i]]
                decision_trees[i].value = threshold[i]

                if verbose:
                    print("%snode=%s test node: go to node %s if %s %s <= %s %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         bcolors.BOLD,
                         self.features[feature[i]],
                         threshold[i],
                         bcolors.ENDC,
                         children_right[i],
                         ))
        return decision_trees[0]


# outlook = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]*1)
# temp = np.asarray([75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70]*1)
# humidity = np.asarray([70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96]*1)
# windy = np.asarray([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]*1)
#
# play = np.asarray([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]*1)
#
# feature_vectors_df = DataFrame()
# feature_vectors_df['outlook'] = outlook
# feature_vectors_df['temp'] = temp
# feature_vectors_df['humidity'] = humidity
# feature_vectors_df['windy'] = windy
#
# labels_df = DataFrame()
# labels_df['cat'] = play

# Read csv into pandas frame
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('../heart.dat', sep=' ')
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
df.columns = columns

# Seperate the dataframe into a class dataframe and feature dataframe
labels_df = DataFrame()
labels_df['cat'] = df['disease']
df = df.drop('disease', axis=1)
feature_vectors_df = df.copy()

tree_constructor = CARTconstructor()
# tree = tree_constructor.construct_tree(feature_vectors_df, labels_df, np.argmax(np.bincount(play)))
# tree.visualise('../tree')

number_of_folds = 2
sum_error_rate = 0.0
total_error_rate = -1
kf = tree_constructor.cross_validation(feature_vectors_df, number_of_folds)

i = 0
for train, test in kf:
    train_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=train)
    test_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=test)
    train_labels_df = DataFrame(labels_df, index=train)
    test_labels_df = DataFrame(labels_df, index=test)

    tree_constructor.construct_tree(train_feature_vectors_df.copy(), train_labels_df)
    # tree_constructor.visualize_tree(tree_constructor.features, labels_df[['cat']], "tree" + str(i))
    # tree_constructor.printTree()
    own_decision_tree = tree_constructor.convertToTree()
    # own_decision_tree.visualise(output_path="../boom"+str(i))
    print "Prediction accuracy rate: %s" % str(1-tree_constructor.calculate_error_rate(tree_constructor, test_feature_vectors_df, test_labels_df))
    sum_error_rate += tree_constructor.calculate_error_rate(tree_constructor, test_feature_vectors_df, test_labels_df)
    # own_decision_tree.to_string()
    # predicted_labels = [None]*len(train_labels_df.index)
    predicted_labels = own_decision_tree.evaluate_multiple(test_feature_vectors_df)
    # for barf in range(len(train_labels_df.index)):
    #     own_decision_tree.
    own_decision_tree.plot_confusion_matrix(test_labels_df['cat'], predicted_labels)

    i += 1
    print "\n\n-------------------------------\n\n"

print "\n\n\n\n\n\nTotal accuracy rate with %d folds: %s" % (number_of_folds, str(1 - sum_error_rate/number_of_folds))

"""
train_feature_vectors_df = DataFrame(feature_vectors_df, index=)

input_vector = DataFrame()
input_vector['outlook'] = np.asarray([1])
input_vector['temp'] = np.asarray([69])
input_vector['humidity'] = np.asarray([97])
input_vector['windy'] = np.asarray([0])
print(tree.evaluate(input_vector))
"""


# TODO: predict probabilities: http://aaaipress.org/Papers/Workshops/2006/WS-06-06/WS06-06-005.pdf
# TODO                         http://cseweb.ucsd.edu/~elkan/calibrated.pdf

# TODO: pruning

# TODO: multivariate splits possible? Split on multiple attributes at once (in C4.5)
