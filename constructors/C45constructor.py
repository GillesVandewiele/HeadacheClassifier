import operator
from math import sqrt, log

from pandas import DataFrame, Series, read_csv
from sklearn.cross_validation import KFold

from constructors.treeconstructor import TreeConstructor
import numpy as np
from scipy.stats import norm

from decisiontree import DecisionTree


class C45Constructor(TreeConstructor):
    def __init__(self):
        pass

    def cross_validation(self, data, k):
        return KFold(len(data.index), n_folds=k, shuffle=True)

    def calculate_entropy(self, values_list):
        # Normalize the values by dividing each value by the sum of all values in the list
        normalized_values = np.fromiter(map(lambda x: x / sum(values_list), values_list), dtype=np.float)

        # Calculate the log of the normalized values (negative because these values are in [0,1])

        log_values = np.fromiter((map(lambda x: log(x,2), normalized_values)), dtype=np.float)

        # Take sum of normalized_values * log_values, multiply with (-1) to get positive float
        return -sum(np.multiply(normalized_values, log_values))

    def split_criterion(self, node):
        """
        Calculates information gain ratio (ratio because normal gain tends to have a strong bias in favor of tests with
        many outcomes) for a subtree
        :param node: the node where the information gain needs to be calculated for
        :return: the information gain: information (entropy) in node - sum(weighted information in its children)
        """
        counts_before_split = np.asarray(node.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_count_before_split = sum(counts_before_split)
        info_before_split = self.calculate_entropy(counts_before_split)

        left_counts = np.asarray(node.left.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_left_count = sum(left_counts)
        right_counts = np.asarray(node.right.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_right_count = sum(right_counts)

        # Information gain after split = weighted entropy of left child + weighted entropy of right child
        # weight = number of nodes in child / sum of nodes in left and right child
        info_after_split = total_left_count / total_count_before_split * self.calculate_entropy(left_counts) \
                           + total_right_count / total_count_before_split * self.calculate_entropy(right_counts)

        return (info_before_split - info_after_split) / info_before_split

    def get_possible_split_values(self, feature_values_cats):
        """
        :param feature_values_cats: pandas dataframe containing the values of a specific feature (first column)
                                    and category (last) of that record
        :return: split_values: list of floats where the tree can possibly split on
        """
        # TODO: possible optimizations: if value consecutive values have same classes, they shouldn't be a split value
        split_values = []
        unique_values = feature_values_cats.sort_values().unique()
        for i in range(len(unique_values)):
            # C4.5 differs from others in not taking the midpoint,so that values also appear as in the feature vectors
            split_values.append(unique_values[i])
            # split_values.append(unique_values[i] + (unique_values[i + 1] - unique_values[i]) / 2)
        return split_values

    def divide_data(self, data, feature, value):
        """
        Divide the data in two subsets, thanks pandas
        :param data: the dataframe to divide
        :param feature: on which column of the dataframe are we splitting?
        :param value: what threshold do we use to split
        :return: node: initialised decision tree object
        """
        return DecisionTree(left=DecisionTree(data=data[data[feature] <= value]),
                            right=DecisionTree(data=data[data[feature] > value]),
                            label=feature,
                            data=data,
                            value=value)

    def all_feature_vectors_equal(self, training_feature_vectors):
        return len(training_feature_vectors.index) == (training_feature_vectors.duplicated(keep='first').sum() + 1)

    def construct_tree(self, training_feature_vectors, labels, default, max_nr_nodes=1):
        """
        Construct a tree from a given array of feature vectors
        :param training_feature_vectors: a pandas dataframe containing the features
        :param labels: a pandas dataframe containing the labels in the same order
        :return: decision_tree: a DecisionTree object
        """

        cols = training_feature_vectors.columns

        data = DataFrame(training_feature_vectors)
        data['cat'] = labels
        unique_labels = np.unique(labels)

        if len(unique_labels) == 0:
            return DecisionTree(label=default)

        # TODO: this has to improve of course (can get stuck now if data not linearly seperable)
        # Stop criterion (when is a node a decision class)
        if len(unique_labels) <= max_nr_nodes or self.all_feature_vectors_equal(data.drop('cat', axis=1)):
            return DecisionTree(label=np.argmax(np.bincount(unique_labels)))

        # For every possible feature and its possible values: calculate the information gain if we would split
        info_gains = {}
        for feature in cols:
            for value in self.get_possible_split_values(data[feature]):
                node = self.divide_data(data, feature, value)
                info_gains[node] = self.split_criterion(node)

        # If info_gains is empty, we have no more possibilities to test, something went wrong
        # Output the default class (this should be adjusted using some background knowledge)
        if len(info_gains) == 0:
            return DecisionTree(label=default)

        # Pick the (feature, value) combination with the highest information gain and create a new node
        best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
        node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)

        # Recursive call to set the left and right child of the newly created node
        node.left = self.construct_tree(best_split_node.left.data.drop('cat', axis=1),
                                        best_split_node.left.data[['cat']], default)
        node.right = self.construct_tree(best_split_node.right.data.drop('cat', axis=1),
                                         best_split_node.right.data[['cat']], default)

        return node

    def calculate_error_rate(self, tree, testing_feature_vectors, labels, significance):
        # If we replace a node by a leaf with its most occuring class, what would then be the error rate?
        #evaluations = tree.evaluate_multiple(testing_feature_vectors)
        if tree.value is None:
            print(labels.values.flatten())
        #if evaluations.shape[0] == 0:
        #    return 0.0
        if len(labels.index) == 0:
            return 0.0
        else:
            error_count = len(labels.index) - np.max(np.bincount(labels.values.flatten()))

            # From count to rate, doing +1 to avoid division by zero
            error_rate = error_count / (len(labels.index)+1)

            if tree.value is None:
                print("halloooo", error_rate * norm.cdf(significance/2) * sqrt((error_rate * (1 - error_rate)) / (len(labels.index)+1)))

            # error rate = |S| * e(T, S) * Z_(alpha/2) * sqrt((e(T,S) * (1-e(T,S)))/|S|)
            return error_rate * norm.cdf(significance/2) * sqrt((error_rate * (1 - error_rate)) / (len(labels.index)+1))
        """
        # Calculate the differences between predicted and correct classes
        # labels is a pandas dataframe, having x rows and just 1 column (the label)
        differences = tree.evaluate_multiple(testing_feature_vectors) - labels.values[:, 0]

        # If the difference between classes is not zero, the predictor made an error
        error_count = np.count_nonzero(differences)

        # From count to rate, doing +1 to avoid division by zero
        error_rate = error_count / (len(labels.values)+1)

        # error rate = |S| * e(T, S) * Z_(alpha/2) * sqrt((e(T,S) * (1-e(T,S)))/|S|)
        return error_rate * norm.cdf(significance/2) * sqrt((error_rate * (1 - error_rate)) / (len(labels.values)+1))
        """

    def set_error_rate(self, tree, testing_feature_vectors, labels):
        """
        # Calculate the differences between predicted and correct classes
        # labels is a pandas dataframe, having x rows and just 1 column (the label)
        differences = tree.evaluate_multiple(testing_feature_vectors) - labels.values[:, 0]

        # If the difference between classes is not zero, the predictor made an error
        error_count = np.count_nonzero(differences)

        """
        # evaluations = tree.evaluate_multiple(testing_feature_vectors)
        #if evaluations.shape[0] == 0:
        #    return 0.0
        #error_count = evaluations.shape[0] - np.max(np.bincount(evaluations))
        if len(labels.index) == 0:
            return 0.0

        error_count = len(labels.index) - np.max(np.bincount(labels.values.flatten()))

        # Calculate the error_rate
        error_rate = error_count / (len(labels.values)+1)
        # Set the pruning ratio of our tree (0.69 is the standard normal distribution of alpha = 0.25)
        tree.pruning_ratio = error_rate * norm.cdf(0.125/2) * sqrt((error_rate * (1 - error_rate)) / (len(labels.values)+1))

        # If we're not in a leaf, we divide the test data and do a recursive call
        if tree.value is not None:
            data = DataFrame(testing_feature_vectors)
            data['cat'] = labels
            node = self.divide_data(data, tree.label, tree.value)
            self.set_error_rate(tree.left, node.left.data.drop('cat', axis=1), node.left.data[['cat']])
            self.set_error_rate(tree.right, node.right.data.drop('cat', axis=1), node.right.data[['cat']])

    def prune_node(self, tree, testing_feature_vectors):
        if tree.value is not None:
            if tree.pruning_ratio < (tree.right.pruning_ratio + tree.left.pruning_ratio):
                # Pick the most occuring class as new class of the leaf
                new_class = np.argmax(np.bincount(tree.evaluate_multiple(tree.data)))
                tree.value = None
                tree.label = new_class
                tree.left = None
                tree.right = None
            else:
                self.prune_node(tree.left, testing_feature_vectors)
                self.prune_node(tree.right, testing_feature_vectors)

    def post_prune(self, tree, testing_feature_vectors, labels, significance=0.125):
        # if tree.left and tree.right are leafs:
        #   calculate error rate of tree and his two children
        # else:
        #   recursive call to calculate error rate of subtrees left and right, eventually proning them

        # If the tree value is None, we are in a leaf, just calculate the error rate and return it
        error_rate = self.calculate_error_rate(tree, testing_feature_vectors, labels, significance)
        if tree.value is None:
            return error_rate
        else:
            # Else, we will need to split up the data for further recursive calls
            data = DataFrame(testing_feature_vectors)
            data['cat'] = labels
            node = self.divide_data(data, tree.label, tree.value)
            # Already calculate the error rate recursively
            left_child_error_rate = self.post_prune(tree.left, node.left.data.drop('cat', axis=1),
                                                    node.left.data[['cat']], significance)
            right_child_error_rate = self.post_prune(tree.right, node.right.data.drop('cat', axis=1),
                                                        node.right.data[['cat']], significance)
            print("error rate = ", error_rate)
            print("left child error = ", left_child_error_rate)
            print("right child error = ", right_child_error_rate)

            if error_rate < (left_child_error_rate + right_child_error_rate)/2:
                new_class = np.argmax(np.bincount(tree.evaluate_multiple(tree.data)))
                tree.value = None
                tree.label = int(new_class)
                tree.left = None
                tree.right = None

                return error_rate

            else:
                return (left_child_error_rate + right_child_error_rate)/2
            """
            # If the left and right children their values are None, they are both leafs
            if tree.left.value is None and tree.right.value is None:
                # Prune if there is an improvement in error rate, the tree becomes a leaf
                if error_rate < (left_child_error_rate + right_child_error_rate):
                    # The new class is equal to the most occurring class in the data
                    new_class = np.argmax(np.bincount(tree.evaluate_multiple(tree.data)))
                    tree.value = None
                    tree.label = int(new_class)
                    tree.left = None
                    tree.right = None

                    # Recalculate the error rate and return it
                    return self.calculate_error_rate(tree, testing_feature_vectors, labels, significance)
                else:
                    return left_child_error_rate + right_child_error_rate

            # Else, one of the children is still a subtree, pruning is different
            else:
                if error_rate < (left_child_error_rate + right_child_error_rate) \
                        and len(node.left.data.index) > len(node.right.data.index):
                    new_tree = tree.left
                else:
                    new_tree = tree.right

                if error_rate < (left_child_error_rate + right_child_error_rate):
                    tree.value = new_tree.value
                    tree.label = new_tree.label
                    tree.left = new_tree.left
                    tree.right = new_tree.right
                    return self.calculate_error_rate(tree, testing_feature_vectors, labels, significance)
                else:
                    return left_child_error_rate + right_child_error_rate
            """

columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('../heart.dat', sep=' ')
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
df.columns = columns

labels_df = DataFrame()
labels_df['cat'] = df['disease']
df = df.drop('disease', axis=1)
feature_vectors_df = df.copy()
print(feature_vectors_df)

tree_constructor = C45Constructor()
kf = tree_constructor.cross_validation(feature_vectors_df, 2)

i = 0
for train, test in kf:
    train_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=train)
    test_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=test)
    train_labels_df = DataFrame(labels_df, index=train)
    test_labels_df = DataFrame(labels_df, index=test)
    decision_tree = tree_constructor.construct_tree(feature_vectors_df.copy(), labels_df, default=0)
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree' + str(i), with_pruning_ratio=True)
    frame = DataFrame(test_feature_vectors_df.copy())
    frame['cat'] = test_labels_df.copy()
    print(frame)
    tree_constructor.post_prune(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree_pruned' + str(i), with_pruning_ratio=True)

"""
outlook = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0]*4)
temp = np.asarray([75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70, 75]*4)
humidity = np.asarray([70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96, 70]*4)
windy = np.asarray([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]*4)

play = np.asarray([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]*4)

feature_vectors_df = DataFrame()
feature_vectors_df['outlook'] = outlook
feature_vectors_df['temp'] = temp
feature_vectors_df['humidity'] = humidity
feature_vectors_df['windy'] = windy

labels_df = DataFrame()
labels_df['cat'] = play

tree_constructor = C45Constructor()
# tree = tree_constructor.construct_tree(feature_vectors_df, labels_df, np.argmax(np.bincount(play)))
# tree.visualise('../tree')

kf = tree_constructor.cross_validation(feature_vectors_df, 2)

i = 0
for train, test in kf:
    train_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=train)
    test_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=test)
    train_labels_df = DataFrame(labels_df, index=train)
    test_labels_df = DataFrame(labels_df, index=test)
    decision_tree = tree_constructor.construct_tree(feature_vectors_df.copy(), labels_df, np.argmax(np.bincount(play)))
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree' + str(i), with_pruning_ratio=True)
    frame = DataFrame(test_feature_vectors_df.copy())
    frame['cat'] = test_labels_df.copy()
    print(frame)
    tree_constructor.post_prune(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree_pruned' + str(i), with_pruning_ratio=True)
    print(i)
    i += 1
"""
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
