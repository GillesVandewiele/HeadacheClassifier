import numpy as np

import sklearn
from pandas import DataFrame
from sklearn.cross_validation import KFold

from constructors.treeconstructor import TreeConstructor
from decisiontree import DecisionTree


class CARTconstructor(TreeConstructor):
    def split_criterion(self, node):
        raise NotImplementedError("This method is not implemented, because we use the optimised sklearn pruning algorithm")

    def __init__(self):
        pass

    def cross_validation(self, data, k):
        return KFold(len(data.index), n_folds=k, shuffle=True)

    def divide_data(self, data, feature, value):
        raise NotImplementedError("This method is not to be used in the cart constructor, because sklearn optimizes this automatically")

    def all_feature_vectors_equal(self, training_feature_vectors):
        return len(training_feature_vectors.index) == (training_feature_vectors.duplicated(keep='first').sum() + 1)

    def anova_f_test(self):
        raise NotImplementedError("This method needs to be implemented")

    def pearson_chi_square_test(self):
        raise NotImplementedError("This method needs to be implemented")

    def construct_tree(self, training_feature_vectors, labels, default, max_nr_nodes=1, discrete_thresh=5):
        """
        Construct a tree from a given array of feature vectors
        :param discrete_thresh:
        :param max_nr_nodes:
        :param default:
        :param training_feature_vectors: a pandas dataframe containing the features
        :param labels: a pandas dataframe containing the labels in the same order
        :return: decision_tree: a DecisionTree object
        """
        cols = training_feature_vectors.columns

        data = DataFrame(training_feature_vectors)
        data['cat'] = labels
        unique_labels = np.unique(labels)

        # Calculate split feature
        feature_p_values = {}
        for feature in cols:
            if len(data[feature].values) > discrete_thresh:
                # Continuous variable, perform ANOVA F test
                feature_p_values[feature] = self.anova_f_test()
            else:
                # Discrete variable, perform Pearson's chi-square test
                feature_p_values[feature] = self.pearson_chi_square_test()


    def calculate_error_rate(self, tree, testing_feature_vectors, labels, significance):
        pass

    def post_prune(self, tree, testing_feature_vectors, labels, significance=0.125):
        pass


outlook = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]*1)
temp = np.asarray([75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70]*1)
humidity = np.asarray([70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96]*1)
windy = np.asarray([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]*1)

play = np.asarray([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]*1)

feature_vectors_df = DataFrame()
feature_vectors_df['outlook'] = outlook
feature_vectors_df['temp'] = temp
feature_vectors_df['humidity'] = humidity
feature_vectors_df['windy'] = windy

labels_df = DataFrame()
labels_df['cat'] = play

tree_constructor = CARTconstructor()
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
    """
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
