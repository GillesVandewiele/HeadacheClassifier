from pandas import read_csv, DataFrame

import operator
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.decisiontree import DecisionTree


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



np.random.seed(84846513)    # 84846513



# Read csv into pandas frame
# columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
# df = read_csv(os.path.join(os.path.join('..', 'data'), 'car.data'), sep=',')
df.columns = columns
# mapping_buy_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
# mapping_doors = {'2': 0, '3': 1, '4': 2, '5more': 3}
# mapping_persons = {'2': 0, '4': 1, 'more': 2}
# mapping_lug = {'small': 0, 'med': 1, 'big': 2}
# mapping_safety = {'low': 0, 'med': 1, 'high': 2}
# mapping_class = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
# df['maint'] = df['maint'].map(mapping_buy_maint)
# df['buying'] = df['buying'].map(mapping_buy_maint)
# df['doors'] = df['doors'].map(mapping_doors)
# df['persons'] = df['persons'].map(mapping_persons)
# df['lug_boot'] = df['lug_boot'].map(mapping_lug)
# df['safety'] = df['safety'].map(mapping_safety)
# df['class'] = df['class'].map(mapping_class)
# permutation = np.random.permutation(df.index)
# df = df.reindex(permutation)
# df = df.reset_index(drop=True)
# df = df.head(300)

features_column_names = ['thal', 'chest pain type', 'number of vessels', 'max heartrate', 'age', 'oldpeak']
column_names = ['thal', 'chest pain type', 'number of vessels', 'max heartrate', 'age', 'oldpeak', 'disease']
df = df[column_names]
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
features_column_names = features_df.columns

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
cart = CARTConstructor()
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
    tree.visualise(os.path.join(os.path.join('..', 'data'), tree_constructor.get_name()))
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


new_tree = merger.regions_to_tree(train_features_df, train_labels_df, merged_regions, features_column_names, feature_mins, feature_maxs)
new_tree.visualise(os.path.join(os.path.join('..', 'data'), 'new_tree'))

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
    for i in range(len(tree_confusion_matrices[tree])):
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
