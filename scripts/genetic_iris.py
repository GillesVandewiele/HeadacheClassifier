from pandas import read_csv, DataFrame

import operator
import os

import re
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
from sklearn import datasets

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.featuredescriptors import DISCRETE, CONTINUOUS

SEED = 1337
N_FOLDS = 3

np.random.seed(SEED)    # 84846513
iris = datasets.load_iris()
df = DataFrame(iris.data)
df.columns = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
df["Name"] = np.add(iris.target, 1)

feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['Name']))

for feature in feature_column_names:
        feature_mins[feature] = np.min(df[feature])
        feature_maxs[feature] = np.max(df[feature])
df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['Name'].copy()
features_df = df.copy()
features_df = features_df.drop('Name', axis=1)
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=1.0)
cart = CARTConstructor(max_depth=5, min_samples_leaf=2)
quest = QuestConstructor(default=1, max_nr_nodes=2, discrete_thresh=1, alpha=0.0000001)
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


    merger = DecisionTreeMerger()
    best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=10,
                                         num_mutations=8, population_size=16, max_samples=2, val_fraction=0.24,
                                         num_boosts=11)
    # best_tree.visualise(os.path.join(os.path.join('..', 'data'), 'best_tree'))
    predicted_labels = best_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Genetic"].append(best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                              predicted_labels.astype(str)))
    #raw_input("Press Enter to continue...")
tree_confusion_matrices_mean = {}

for key in tree_confusion_matrices:
    print key
    for matrix in tree_confusion_matrices[key]:
        print matrix

fig = plt.figure()
fig.suptitle('Accuracy on iris dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
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