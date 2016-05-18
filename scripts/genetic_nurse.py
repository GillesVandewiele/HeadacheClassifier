"""
parents:     usual, pretentious, great_pret.
has_nurs:    proper, less_proper, improper, critical, very_crit.
form:        complete, completed, incomplete, foster.
children:    1, 2, 3, more.
housing:     convenient, less_conv, critical.
finance:     convenient, inconv.
social:      nonprob, slightly_prob, problematic.
health:      recommended, priority, not_recom.
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
N_FOLDS = 5

np.random.seed(SEED)    # 84846513
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

mapping_parents = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
mapping_has_nurs = {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4}
mapping_form = {'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3}
mapping_housing = {'convenient': 0, 'less_conv': 1, 'critical': 2}
mapping_finance = {'convenient': 0, 'inconv': 1}
mapping_social = {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2}
mapping_health = {'recommended': 0, 'priority': 1, 'not_recom': 2}
mapping_class = {'not_recom': 1, 'recommend': 0, 'very_recom': 2, 'priority': 3, 'spec_prior': 4}

df = read_csv(os.path.join('../data', 'nursery.data'), sep=',')
df = df.dropna()
df.columns=columns

df['parents'] = df['parents'].map(mapping_parents)
df['has_nurs'] = df['has_nurs'].map(mapping_has_nurs)
df['form'] = df['form'].map(mapping_form)
df['children'] = df['children'].map(lambda x: 4 if x == 'more' else int(x))
df['housing'] = df['housing'].map(mapping_housing)
df['finance'] = df['finance'].map(mapping_finance)
df['social'] = df['social'].map(mapping_social)
df['health'] = df['health'].map(mapping_health)
df['class'] = df['class'].map(mapping_class)

df = df[df['class'] != 0]
df = df.reset_index(drop=True)
#
# print df
# print np.bincount(df['class'])

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
cart = CARTConstructor(min_samples_leaf=40, max_depth=10)
quest = QuestConstructor(default=1, max_nr_nodes=50, discrete_thresh=15, alpha=0.001)
tree_constructors = [c45, cart, quest]
# tree_constructors = [c45]

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
        # tree.visualise(os.path.join(os.path.join('..', 'data'), tree_constructor.get_name()))
        trees.append(tree)
        predicted_labels = tree.evaluate_multiple(test_features_df)
        tree_confusion_matrices[tree_constructor.get_name()].append(tree.plot_confusion_matrix(test_labels_df['cat']
                                                                                               .values.astype(str),
                                                                    predicted_labels.astype(str)))


    merger = DecisionTreeMerger()
    best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=5,
                                         num_mutations=3, population_size=6, max_samples=35, val_fraction=0.10,
                                         num_boosts=3)

    predicted_labels = best_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Genetic"].append(best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                              predicted_labels.astype(str)))

tree_confusion_matrices_mean = {}
for key in tree_confusion_matrices:
    print key
    for matrix in tree_confusion_matrices[key]:
        print matrix

fig = plt.figure()
fig.suptitle('Accuracy on nursery dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
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