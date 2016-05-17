"""
Shuttle dataset:
----------------
----------------

9 features:
-----------


7 classes:
-----------
1 Rad Flow
2 Fpv Close
3 Fpv Open
4 High
5 Bypass
6 Bpv Close
7 Bpv Open

Approximately 80% of the data belongs to class 1. Therefore the default
accuracy is about 80%. The aim here is to obtain an accuracy of
99 - 99.9%.
"""

from pandas import read_csv, DataFrame

import operator
import os

import re
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from extractors.featureselector import RF_feature_selection

SEED = 1337
N_FOLDS = 2

np.random.seed(SEED)    # 84846513
columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
           'feature9', 'class' ]
df = read_csv(os.path.join('../data', 'shuttle.tst'), sep=' ')
df.columns=columns
df = df[df['class'] < 6]
df = df.reset_index(drop=True)
print np.bincount(df['class'])
feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['class']))

for feature in feature_column_names:
    if np.min(df[feature]) < 0:
        df[feature] += np.min(df[feature]) * (-1)
        feature_mins[feature] = 0
    else:
        feature_mins[feature] = np.min(df[feature])

    feature_maxs[feature] = np.max(df[feature])

df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['class'].copy()
features_df = df.copy()
features_df = features_df.drop('class', axis=1)
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=0.05)
cart = CARTConstructor(min_samples_leaf=10, max_depth=6)
quest = QuestConstructor(default=1, max_nr_nodes=1, discrete_thresh=25, alpha=0.05)
tree_constructors = [c45, cart, quest]

tree_confusion_matrices = {}
for tree_constructor in tree_constructors:
    tree_confusion_matrices[tree_constructor.get_name()] = []
tree_confusion_matrices["Genetic"] = []

skf = sklearn.cross_validation.StratifiedKFold(labels_df['cat'], n_folds=N_FOLDS, shuffle=False, random_state=SEED)

for train_index, test_index in skf:
    train_features_df, test_features_df = features_df.iloc[train_index,:].copy(), features_df.iloc[test_index,:].copy()
    train_labels_df, test_labels_df = labels_df.iloc[train_index,:].copy(), labels_df.iloc[test_index,:].copy()
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = test_features_df.reset_index(drop=True)
    train_labels_df = train_labels_df.reset_index(drop=True)
    test_labels_df = test_labels_df.reset_index(drop=True)
    train_df = train_features_df.copy()
    train_df['cat'] = train_labels_df['cat'].copy()

    trees = []

    for tree_constructor in tree_constructors:
        tree = tree_constructor.construct_tree(train_features_df, train_labels_df)
        # tree.visualise(os.path.join(os.path.join('..', 'data'), tree_constructor.get_name()))
        trees.append(tree)
        predicted_labels = tree.evaluate_multiple(test_features_df)
        tree_confusion_matrices[tree_constructor.get_name()].append(tree.plot_confusion_matrix(test_labels_df['cat']
                                                                                               .values.astype(str),
                                                                    predicted_labels.astype(str)))


    merger = DecisionTreeMerger()
    best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=10,
                                         num_mutations=5, population_size=10, max_samples=2, val_fraction=0.10,
                                         num_boosts=3)

    # best_tree.visualise(os.path.join(os.path.join('..', 'data'), 'best_tree'))
    predicted_labels = best_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Genetic"].append(best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                              predicted_labels.astype(str)))

tree_confusion_matrices_mean = {}
for key in tree_confusion_matrices:
    print key
    for matrix in tree_confusion_matrices[key]:
        print matrix

fig = plt.figure()
fig.suptitle('Accuracy on heart disease dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
counter = 0
for key in tree_confusion_matrices:
    tree_confusion_matrices_mean[key] = np.zeros(tree_confusion_matrices[key][0].shape)
    for i in range(len(tree_confusion_matrices[key])):
        tree_confusion_matrices_mean[key] = np.add(tree_confusion_matrices_mean[key], tree_confusion_matrices[key][i])
    cm_normalized = np.around(tree_confusion_matrices_mean[key].astype('float') / tree_confusion_matrices_mean[key].sum(axis=1)[:, np.newaxis], 4)

    diagonal_sum = sum([tree_confusion_matrices_mean[key][i][i] for i in range(len(tree_confusion_matrices_mean[key]))])
    total_count = np.sum(tree_confusion_matrices_mean[key])
    print tree_confusion_matrices_mean[key], float(diagonal_sum)/float(total_count)

    ax = fig.add_subplot(1, len(tree_confusion_matrices), counter+1)
    cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_title(key, y=1.08)
    for (j,i),label in np.ndenumerate(cm_normalized):
        ax.text(i,j,label,ha='center',va='center')
    if counter == len(tree_confusion_matrices)-1:
        fig.colorbar(cax,fraction=0.046, pad=0.04)
    counter += 1

F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1], forward=True)
plt.show()