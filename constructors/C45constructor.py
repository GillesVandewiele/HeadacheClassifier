import operator
from math import log2

from pandas import DataFrame, Series

from constructors.treeconstructor import TreeConstructor
import numpy as np

from decisiontree import DecisionTree


class C45Constructor(TreeConstructor):
    def __init__(self):
        pass

    def calculate_entropy(self, values_list):
        # Normalize the values by dividing each value by the sum of all values in the list
        normalized_values = np.fromiter(map(lambda x: x / sum(values_list), values_list), dtype=np.float)

        # Calculate the log of the normalized values (negative because these values are in [0,1])
        log_values = np.fromiter((map(lambda x: log2(x), normalized_values)), dtype=np.float)

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

        return (info_before_split - info_after_split)/info_before_split

    def get_possible_split_values(self, feature_values):
        """
        :param feature_values: pandas dataframe containing the values of a specific feature and the class of that record
        :return: split_values: list of floats where the tree can possibly split on
        """
        # TODO: possible optimizations: if value consecutive values have same classes, they shouldn't be a split value
        split_values = []
        unique_values = feature_values.sort_values().unique()
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

    def construct_tree(self, feature_vectors, labels, default):
        """
        Construct a tree from a given array of feature vectors
        :param feature_vectors: a pandas dataframe containing the features
        :param labels: a pandas dataframe containing the labels in the same order
        :return: decision_tree: a DecisionTree object
        """

        cols = feature_vectors.columns

        data = DataFrame(feature_vectors)
        data['cat'] = labels
        unique_labels = np.unique(labels)

        # TODO: this has to improve of course (can get stuck now if data not linearly seperable)
        # Stop criterion (when is a node a decision class)
        if len(unique_labels) == 1:
            return DecisionTree(label=unique_labels[0])

        # For every possible feature and its possible values: calculate the information gain if we would split
        info_gains = {}
        for feature in cols:
            for value in self.get_possible_split_values(data[feature]):
                node = self.divide_data(data, feature, value)
                info_gains[node] = self.split_criterion(node)

        # If info_gains is empty, we have no more possibilities to test, something went wrong
        # Output the default class (this should be adjusted using some background knowledge)
        if len(info_gains) == 0:
            return DecisionTree(label=unique_labels[0])

        # Pick the (feature, value) combination with the highest information gain and create a new node
        best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
        node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)

        # Recursive call to set the left and right child of the newly created node
        node.left = self.construct_tree(best_split_node.left.data.drop('cat', axis=1),
                                        best_split_node.left.data[['cat']], default)
        node.right = self.construct_tree(best_split_node.right.data.drop('cat', axis=1),
                                         best_split_node.right.data[['cat']], default)

        return node


outlook = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
temp = np.asarray([75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70])
humidity = np.asarray([70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96])
windy = np.asarray([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0])

play = np.asarray([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1])

feature_vectors_df = DataFrame()
feature_vectors_df['outlook'] = outlook
feature_vectors_df['temp'] = temp
feature_vectors_df['humidity'] = humidity
feature_vectors_df['windy'] = windy

labels_df = DataFrame()
labels_df['cat'] = play

frame = DataFrame(feature_vectors_df.copy())
frame['cat'] = labels_df.copy()
print(frame)

tree_constructor = C45Constructor()
tree = tree_constructor.construct_tree(feature_vectors_df, labels_df, np.argmax(np.bincount(play)))
tree.visualise('../tree')

input_vector = DataFrame()
input_vector['outlook'] = np.asarray([1])
input_vector['temp'] = np.asarray([69])
input_vector['humidity'] = np.asarray([97])
input_vector['windy'] = np.asarray([0])
print(tree.evaluate(input_vector))