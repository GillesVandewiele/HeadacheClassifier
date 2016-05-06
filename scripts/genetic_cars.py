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
from objects.featuredescriptors import DISCRETE, CONTINUOUS

SEED = 1337
N_FOLDS = 5

np.random.seed(SEED)    # 84846513
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'car.data'), sep=',')
df.columns=columns
mapping_buy_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
mapping_doors = {'2': 0, '3': 1, '4': 2, '5more': 3}
mapping_persons = {'2': 0, '4': 1, 'more': 2}
mapping_lug = {'small': 0, 'med': 1, 'big': 2}
mapping_safety = {'low': 0, 'med': 1, 'high': 2}
mapping_class = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

df['maint'] = df['maint'].map(mapping_buy_maint)
df['buying'] = df['buying'].map(mapping_buy_maint)
df['doors'] = df['doors'].map(mapping_doors)
df['persons'] = df['persons'].map(mapping_persons)
df['lug_boot'] = df['lug_boot'].map(mapping_lug)
df['safety'] = df['safety'].map(mapping_safety)
df['class'] = df['class'].map(mapping_class).astype(int)

# perm = np.random.permutation(df.index)
# df = df.reindex(perm)
# df = df.tail(0.5*len(df))
# df = df.reset_index(drop=True)

feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['class']))

for feature in feature_column_names:
        feature_mins[feature] = np.min(df[feature])
        feature_maxs[feature] = np.max(df[feature])
df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['class'].copy()
features_df = df.copy()
features_df = features_df.drop('class', axis=1)
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=0.95)
cart = CARTConstructor(max_depth=12, min_samples_leaf=2)
quest = QuestConstructor(default=1, max_nr_nodes=1, discrete_thresh=10, alpha=0.99)
# c45 = C45Constructor(cf=0.75)
# cart = CARTConstructor(max_depth=10, min_samples_leaf=2)
# quest = QuestConstructor(default=1, max_nr_nodes=2, discrete_thresh=10, alpha=0.9)
tree_constructors = [c45, cart, quest]

tree_confusion_matrices = {}
for tree_constructor in tree_constructors:
    tree_confusion_matrices[tree_constructor.get_name()] = []
tree_confusion_matrices["Genetic"] = []

skf = sklearn.cross_validation.StratifiedKFold(labels_df['cat'], n_folds=N_FOLDS, shuffle=True, random_state=SEED)

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
        #tree.visualise(os.path.join(os.path.join('..', 'data'), tree_constructor.get_name()))
        trees.append(tree)
        predicted_labels = tree.evaluate_multiple(test_features_df)
        tree_confusion_matrices[tree_constructor.get_name()].append(tree.plot_confusion_matrix(test_labels_df['cat']
                                                                                               .values.astype(str),
                                                                    predicted_labels.astype(str)))
        print tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))


    merger = DecisionTreeMerger()
    best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=5,
                                         num_mutations=3, population_size=6, max_samples=2, val_fraction=0.25,
                                         num_boosts=3)
    #best_tree.visualise(os.path.join(os.path.join('..', 'data'), 'best_tree'))
    predicted_labels = best_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Genetic"].append(best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                              predicted_labels.astype(str)))
    print best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))
    #raw_input("Press Enter to continue...")

tree_confusion_matrices_mean = {}
for key in tree_confusion_matrices:
    print key
    for matrix in tree_confusion_matrices[key]:
        print matrix

fig = plt.figure()
fig.suptitle('Accuracy on cars dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
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