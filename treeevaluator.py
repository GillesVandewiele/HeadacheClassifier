import random
from pandas import read_csv, DataFrame

import operator

import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from constructors.CARTconstructor import CARTconstructor
from constructors.questconstructor import QuestConstructor
from constructors.C45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from decisiontree import DecisionTree


class TreeEvaluator(object):

    def __init__(self):
        pass

    def evaluate_trees(self, data, tree_constructors, n_folds=2):
        kf = sklearn.cross_validation.KFold(len(data.index), n_folds=n_folds)
        tree_confusion_matrices = {}
        labels_df = DataFrame()
        labels_df['cat'] = data['disease'].copy()
        data = data.drop('disease', axis=1)
        feature_vectors_df = data.copy()
        for train, test in kf:
            X_train = DataFrame(feature_vectors_df, index=train)
            X_test = DataFrame(feature_vectors_df, index=test)
            y_train = DataFrame(labels_df, index=train)
            y_test = DataFrame(labels_df, index=test)
            for tree_constructor in tree_constructors:
                tree = tree_constructor.construct_tree(X_train, y_train)
                tree.visualise(tree_constructor.get_name())
                predicted_labels = tree.evaluate_multiple(X_test)
                print tree_constructor.get_name(), predicted_labels
                if tree_constructor not in tree_confusion_matrices:
                    tree_confusion_matrices[tree_constructor] = [tree.plot_confusion_matrix(y_test['cat'].values.astype(str), predicted_labels)]
                else:
                    tree_confusion_matrices[tree_constructor].append(tree.plot_confusion_matrix(y_test['cat'].values.astype(str), predicted_labels))

        fig = plt.figure()
        tree_confusion_matrices_mean = {}
        counter = 1
        for tree_constructor in tree_constructors:
            tree_confusion_matrices_mean[tree_constructor] = np.zeros(tree_confusion_matrices[tree_constructor][0].shape)
            for i in range(n_folds):
                tree_confusion_matrices_mean[tree_constructor] = np.add(tree_confusion_matrices_mean[tree_constructor], tree_confusion_matrices[tree_constructor][i])
            tree_confusion_matrices[tree_constructor] = np.divide(tree_confusion_matrices_mean[tree_constructor], len(tree_confusion_matrices[tree_constructor]))
            tree_confusion_matrices[tree_constructor] = np.divide(tree_confusion_matrices_mean[tree_constructor], np.matrix.sum(np.asmatrix(tree_confusion_matrices_mean[tree_constructor]))).round(3)

            ax = fig.add_subplot(len(tree_constructors), 1, counter)
            cax = ax.matshow(tree_confusion_matrices[tree_constructor], cmap=plt.get_cmap('RdYlGn'))
            ax.set_title(tree_constructor.get_name())
            for (j,i),label in np.ndenumerate(tree_confusion_matrices[tree_constructor]):
                ax.text(i,j,label,ha='center',va='center')
            fig.colorbar(cax)
            counter += 1

        pl.show()

# Read csv into pandas frame
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('heart.dat', sep=' ')
#df = df.iloc[np.random.permutation(len(df))]
#df = df.reset_index(drop=True)
df.columns = columns

#features_column_names = ['max heartrate', 'resting blood pressure', 'serum cholestoral', 'oldpeak']
features_column_names = ['oldpeak', 'max heartrate', 'resting blood pressure', 'serum cholestoral']
# labels_column_names = 'disease'
column_names = ['oldpeak', 'max heartrate', 'resting blood pressure', 'serum cholestoral', 'disease']
df = df[column_names]
# df = df.drop(columns[:3], axis=1)
# df = df.drop(columns[4:7], axis=1)
# df = df.drop(columns[8:-1], axis=1)
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
features_column_names = features_df.columns

np.random.seed(2355838997)
permutation = np.random.permutation(features_df.index)
features_df = features_df.reindex(permutation)
features_df = features_df.reset_index(drop=True)
labels_df = labels_df.reindex(permutation)
labels_df = labels_df.reset_index(drop=True)

train_features_df = features_df.head(int(0.8*len(features_df.index)))
test_features_df = features_df.tail(int(0.2*len(features_df.index)))
train_labels_df = labels_df.head(int(0.8*len(labels_df.index)))
test_labels_df = labels_df.tail(int(0.2*len(labels_df.index)))


c45 = C45Constructor()
cart = CARTconstructor()
quest = QuestConstructor()
tree_constructors = [c45, cart, quest]
evaluator = TreeEvaluator()
#evaluator.evaluate_trees(df, tree_constructors)
merger = DecisionTreeMerger()
regions_list = []
constructed_trees = []
for tree_constructor in tree_constructors:
    tree = tree_constructor.construct_tree(train_features_df, train_labels_df)
    tree.populate_samples(train_features_df, train_labels_df['cat'])
    tree.visualise(tree_constructor.get_name())
    regions = merger.decision_tree_to_decision_table(tree, train_features_df)
    regions_list.append(regions)
    constructed_trees.append(tree)
    # merger.plot_regions("rect_"+tree_constructor.get_name()+".png", regions, ['1', '2'], features_column_names[0],
    #                     features_column_names[1], x_max=np.max(features_df[features_column_names[0]].values),
    #                     y_max=np.max(features_df[features_column_names[1]].values),
    #                     x_min=np.min(features_df[features_column_names[0]].values),
    #                     y_min=np.min(features_df[features_column_names[1]].values))
feature_mins = {}
feature_maxs = {}
for feature in features_column_names:
    feature_mins[feature] = np.min(train_features_df[feature])
    feature_maxs[feature] = np.max(train_features_df[feature])
merged_regions = merger.calculate_intersection(regions_list[0], regions_list[2], features_column_names, feature_maxs,
                                               feature_mins)
merged_regions = merger.calculate_intersection(merged_regions, regions_list[1], features_column_names, feature_maxs,
                                              feature_mins)
# merger.plot_regions("intersected.png", merged_regions, ['1', '2'], features_column_names[0],
#                     features_column_names[1], x_max=np.max(features_df[features_column_names[0]].values),
#                     y_max=np.max(features_df[features_column_names[1]].values),
#                     x_min=np.min(features_df[features_column_names[0]].values),
#                     y_min=np.min(features_df[features_column_names[1]].values))

def calculate_entropy(values_list):
        if sum(values_list) == 0:
            return 0
        # Normalize the values by dividing each value by the sum of all values in the list
        normalized_values = map(lambda x: float(x) / float(sum(values_list)), values_list)

        # Calculate the log of the normalized values (negative because these values are in [0,1])

        log_values = map(lambda x: np.log(x)/np.log(2), normalized_values)

        # Take sum of normalized_values * log_values, multiply with (-1) to get positive float
        return -sum(np.multiply(normalized_values, log_values))


def split_criterion(node):
        """
        Calculates information gain ratio (ratio because normal gain tends to have a strong bias in favor of tests with
        many outcomes) for a subtree
        :param node: the node where the information gain needs to be calculated for
        :return: the information gain: information (entropy) in node - sum(weighted information in its children)
        """
        counts_before_split = np.asarray(node.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_count_before_split = sum(counts_before_split)
        info_before_split = calculate_entropy(counts_before_split)

        if len(node.left.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            left_counts = np.asarray(node.left.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            left_counts = [0]
        total_left_count = sum(left_counts)
        if len(node.right.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            right_counts = np.asarray(node.right.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            right_counts = [0]
        total_right_count = sum(right_counts)

        # Information gain after split = weighted entropy of left child + weighted entropy of right child
        # weight = number of nodes in child / sum of nodes in left and right child
        info_after_split = float(total_left_count) / float(total_count_before_split) * calculate_entropy(left_counts) \
                           + float(total_right_count) / float(total_count_before_split) * calculate_entropy(right_counts)

        return (info_before_split - info_after_split) / info_before_split


def divide_data(data, feature, value):
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

def dec(input_, output_):
    if type(input_) is list:
        for subitem in input_:
            dec(subitem, output_)
    else:
        output_.append(input_)


def regions_to_tree(features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=1):
    # Initialize the feature bounds on their mins and maxs
    bounds = {}
    for _feature in features:
        bounds[_feature] = [feature_mins[_feature], feature_maxs[_feature]]

    # For each feature f, we look for a line that for each f' != f goes from bounds[f'][0] to bounds[f'][1]
    lines = {}
    for _feature in features:
        print "checking for lines ", _feature
        connected_lower_upper_regions = {}
        for other_feature in features:  # Complexity O(d^2) already
            if _feature != other_feature:
                # Check if we can find 2 points, where the _feature values are the same, and other_feature values
                # are equal to their required lower and upper bound. Then check if we can draw a line between
                # these two points. We always check the left line of the region, except for the most left regions

                # First find all lower regions: their lower bound is equal to the required lower bound (saved in bounds)
                # and it cannot be the most left line (this is the case when the lower bounds for _feature is equal
                # to its minimum
                lower_regions = []
                for region in regions:  # Complexity O(d^2 * |B|)
                    if region[other_feature][0] == bounds[other_feature][0] \
                            and region[_feature][0] != feature_mins[_feature] \
                            and region[_feature][0] != feature_maxs[_feature]:
                        lower_regions.append(region)

                # Now find upper regions with the same value for _feature as a region in lower_regions
                lower_upper_regions = []
                for region in regions:
                    if region[other_feature][1] == bounds[other_feature][1]:
                        for lower_region in lower_regions:  # Even a little bit more complexity here
                            if lower_region[_feature][0] == region[_feature][0]:
                                lower_upper_regions.append([lower_region, region])

                # Now check if we can draw a line between the lower and upper region
                connected_lower_upper_regions[other_feature] = []
                for lower_upper_region in lower_upper_regions:
                    lowest_upper_bound = lower_upper_region[0][other_feature][1]
                    highest_lower_bound = lower_upper_region[1][other_feature][0]
                    still_searching = True
                    while still_searching:
                        # Line is connected when either the highest lower bound and lowest upper bound
                        # are adjacent (number of regions along the line is even). Or when
                        # both the lower bounds are equal (odd case, handled further)
                        if highest_lower_bound == lowest_upper_bound:
                            connected_lower_upper_regions[other_feature].append(lower_upper_region)
                            break

                        found_new_lower_bound = False
                        found_new_upper_bound = False
                        for region in regions:  # O boy, this complexity is getting out of hand
                            if found_new_upper_bound and found_new_lower_bound:
                                break

                            if region[_feature][0] == lower_upper_region[0][_feature][0] \
                                and region[other_feature][0] == lowest_upper_bound:
                                found_new_upper_bound = True
                                lowest_upper_bound = region[other_feature][1]

                            if region[_feature][0] == lower_upper_region[1][_feature][0] \
                                and region[other_feature][1] == highest_lower_bound:
                                found_new_lower_bound = True
                                if region[other_feature][1] == lowest_upper_bound:  # This is the odd case
                                    connected_lower_upper_regions[other_feature].append(lower_upper_region)
                                highest_lower_bound = region[other_feature][0]

                        still_searching = found_new_lower_bound and found_new_upper_bound

        # Now for all these connected_lower_upper_regions in each dimension, we need to find all line
        # where the value of f is equal (the bounds constraint are already fulfilled)
        if sum([1 if len(value) > 0 else 0 for value in connected_lower_upper_regions.values()]) == len(features)-1:
            # We found a line fulfilling bounds constraints in all other dimensions
            lines[_feature] = []
            temp = []
            dec(connected_lower_upper_regions.values(), temp)
            for region in temp:
                if region[_feature][0] not in lines[_feature]:
                    lines[_feature].append(region[_feature][0])

    print lines

    # When we looped over each possible feature and found each possible split line, we split the data
    # Using the feature and value of the line and pick the best one
    data = DataFrame(features_df)
    data['cat'] = labels_df
    info_gains = {}
    for key in lines:
        for value in lines[key]:
            node = divide_data(data, key, value)
            split_crit = split_criterion(node)
            if split_crit > 0:
                info_gains[node] = split_criterion(node)

    print info_gains

    if len(info_gains) > 0:
        best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
        node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)
    else:
        node = DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), data=data, value=None)
        return node
    print node.label, node.value

    ##########################################################################

    # We call recursively with the splitted data and set the bounds of feature f
    # for left child: set upper bounds to the value of the chosen line
    # for right child: set lower bounds to value of chosen line
    feature_mins_right = feature_mins.copy()
    feature_mins_right[node.label] = node.value
    feature_maxs_left = feature_maxs.copy()
    feature_maxs_left[node.label] = node.value
    if len(best_split_node.left.data) >= max_samples and len(best_split_node.right.data) >= max_samples:
        node.left = regions_to_tree(best_split_node.left.data.drop('cat', axis=1), best_split_node.left.data[['cat']],
                                    regions, features, feature_mins, feature_maxs_left)
        node.right = regions_to_tree(best_split_node.right.data.drop('cat', axis=1), best_split_node.right.data[['cat']],
                                    regions, features, feature_mins_right, feature_maxs)
    else:
        node.label = str(np.argmax(np.bincount(labels_df['cat'].values.astype(int))))
        node.value = None

    return node

new_tree = regions_to_tree(train_features_df, train_labels_df, merged_regions, features_column_names, feature_mins, feature_maxs)
new_tree.visualise("new_tree")

trees = [constructed_trees[0], constructed_trees[1], constructed_trees[2], new_tree]

tree_confusion_matrices = {}
for tree in trees:
    predicted_labels = tree.evaluate_multiple(test_features_df)
    if tree not in tree_confusion_matrices:
        tree_confusion_matrices[tree] = [tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))]
    else:
        tree_confusion_matrices[tree].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))

fig = plt.figure()
tree_confusion_matrices_mean = {}
counter = 1
for tree in trees:
    tree_confusion_matrices_mean[tree] = np.zeros(tree_confusion_matrices[tree][0].shape)
    for i in range(1):
        tree_confusion_matrices_mean[tree] = np.add(tree_confusion_matrices_mean[tree], tree_confusion_matrices[tree][i])
    tree_confusion_matrices[tree] = np.divide(tree_confusion_matrices_mean[tree], len(tree_confusion_matrices[tree]))
    tree_confusion_matrices[tree] = np.divide(tree_confusion_matrices_mean[tree], np.matrix.sum(np.asmatrix(tree_confusion_matrices_mean[tree]))).round(3)

    ax = fig.add_subplot(len(trees), 1, counter)
    cax = ax.matshow(tree_confusion_matrices[tree], cmap=plt.get_cmap('RdYlGn'))
    for (j,i),label in np.ndenumerate(tree_confusion_matrices[tree]):
        ax.text(i,j,label,ha='center',va='center')
    fig.colorbar(cax)
    counter += 1

pl.show()

# We now artifically construct a decision tree in the following way:
# For each region: we pick a sample in the middle of the region,
#                  assign the class with highest prob to it and add it to a list
# def generate_samples(regions, features):
#     _samples = DataFrame()
#     for region in regions:
#         region_samples = []
#         max_side1 = 0
#         max_side2 = 0
#         max_side3 = 0
#         for _feature in features:
#             side = (region[_feature][1] - region[_feature][0])
#             if side > max_side1:
#                 max_side3 = max_side2
#                 max_side2 = max_side1
#                 max_side1 = side
#             elif side > max_side2:
#                 max_side3 = max_side2
#                 max_side2 = side
#             elif side > max_side3:
#                 max_side3 = side
#
#
#         number_of_samples_per_region = int(np.log2((max_side1+1)*(max_side2+1)*(max_side3+1))*pow(np.max(region['class'].values(), 2)))
#         print number_of_samples_per_region
#
#         for k in range(number_of_samples_per_region):
#             region_samples.append({})
#
#         for _feature in features:
#             for index in range(number_of_samples_per_region):
#                 region_samples[index][_feature] = region[_feature][0] + random.random() * \
#                                                                         ((region[_feature][1] - region[_feature][0])+1)
#
#         for sample in region_samples:
#             sample['cat'] = max(region['class'].iteritems(), key=operator.itemgetter(1))[0]
#             _samples = _samples.append(sample, ignore_index=True)
#     return _samples
#
# samples = generate_samples(merged_regions, features_column_names)
# print samples
# sample_labels_df = samples[['cat']]
# sample_features_df = samples.copy()
# sample_features_df = sample_features_df.drop('cat', axis=1)
# new_tree = c45.construct_tree(sample_features_df, sample_labels_df)
# new_tree.populate_samples(train_features_df, train_labels_df['cat'])
# new_tree.visualise("new_tree")
#
# trees = [constructed_trees[0], constructed_trees[1], constructed_trees[2], new_tree]
#
# tree_confusion_matrices = {}
# for tree in trees:
#     predicted_labels = tree.evaluate_multiple(test_features_df)
#     if tree not in tree_confusion_matrices:
#         tree_confusion_matrices[tree] = [tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))]
#     else:
#         tree_confusion_matrices[tree].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))
#
# fig = plt.figure()
# tree_confusion_matrices_mean = {}
# counter = 1
# for tree in trees:
#     tree_confusion_matrices_mean[tree] = np.zeros(tree_confusion_matrices[tree][0].shape)
#     for i in range(1):
#         tree_confusion_matrices_mean[tree] = np.add(tree_confusion_matrices_mean[tree], tree_confusion_matrices[tree][i])
#     tree_confusion_matrices[tree] = np.divide(tree_confusion_matrices_mean[tree], len(tree_confusion_matrices[tree]))
#     tree_confusion_matrices[tree] = np.divide(tree_confusion_matrices_mean[tree], np.matrix.sum(np.asmatrix(tree_confusion_matrices_mean[tree]))).round(3)
#
#     ax = fig.add_subplot(len(trees), 1, counter)
#     cax = ax.matshow(tree_confusion_matrices[tree], cmap=plt.get_cmap('RdYlGn'))
#     for (j,i),label in np.ndenumerate(tree_confusion_matrices[tree]):
#         ax.text(i,j,label,ha='center',va='center')
#     fig.colorbar(cax)
#     counter += 1
#
# pl.show()


# Each of the lines in our rectangle, is a split on a specific feature and specific value
# We now build a tree from these splits:
#   Starting at the root node, we consider each of the splits and pick the best one (impurity metric)
#   Then we remove that split from the set and divide the rectangles in two using that split
#   We call recursively, creating its left and right child



"""
tree_evaluator = TreeEvaluator()
quest = QuestConstructor()
cart = CARTconstructor()
c45 = C45Constructor()
tree_constructors = [quest, cart, c45]
tree_evaluator.evaluate_trees(df, tree_constructors, n_folds=2)
"""
