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
from sklearn.cross_validation import StratifiedKFold

from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.featuredescriptors import DISCRETE, CONTINUOUS

SEED = 1337

iris = datasets.load_iris()
df = DataFrame(iris.data)
df.columns = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
df["Name"] = np.add(iris.target, 1)
descriptors = [(CONTINUOUS,), (CONTINUOUS,), (CONTINUOUS,), (CONTINUOUS,), (DISCRETE, len(np.unique(df['Name'])))]

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
features_df = features_df/features_df.max()
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=0.75)

tree_confusion_matrices = {}
titles = ["Unaugmented C4.5", "Augmented C4.5"]

skf = StratifiedKFold(labels_df['cat'], n_folds=5, shuffle=True, random_state=SEED)

for train_index, test_index in skf:
    train_features_df, test_features_df = features_df.iloc[train_index,:].copy(), features_df.iloc[test_index,:].copy()
    train_labels_df, test_labels_df = labels_df.iloc[train_index,:].copy(), labels_df.iloc[test_index,:].copy()
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = test_features_df.reset_index(drop=True)
    train_labels_df = train_labels_df.reset_index(drop=True)
    test_labels_df = test_labels_df.reset_index(drop=True)
    train_df = train_features_df.copy()
    train_df['cat'] = train_labels_df['cat'].copy()
    tree = c45.construct_tree(train_features_df, train_labels_df)
    tree.populate_samples(train_features_df, train_labels_df['cat'])
    # tree.visualise('c45_unaugmented')
    merger = DecisionTreeMerger()
    regions = merger.decision_tree_to_decision_table(tree, train_features_df)
    for region in regions:
        for feature in feature_column_names:
            if region[feature][0] == float("-inf"):
                region[feature][0] = feature_mins[feature]
            if region[feature][1] == float("inf"):
                region[feature][1] = feature_maxs[feature]
    new_df = merger.generate_samples(regions, features_df.columns, descriptors)
    sample_labels_df = new_df[['cat']].copy()
    augmented_labels_df = sample_labels_df.append(train_labels_df, ignore_index=True)
    new_df = new_df.drop('cat', axis=1)
    augmented_features_df = new_df.append(train_features_df, ignore_index=True)
    augmented_features_df = augmented_features_df.astype(float)
    augmented_tree = c45.construct_tree(augmented_features_df, augmented_labels_df)
    augmented_tree.populate_samples(train_features_df, train_labels_df['cat'])
    # augmented_tree.visualise('c45_augmented')
    trees = {'unaug': tree, 'aug': augmented_tree}
    for tree in trees:
        predicted_labels = trees[tree].evaluate_multiple(test_features_df)
        if tree in tree_confusion_matrices:
            tree_confusion_matrices[tree].append(trees[tree].plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))
        else:
            tree_confusion_matrices[tree] = [trees[tree].plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))]

print tree_confusion_matrices

tree_confusion_matrices_mean = {}
counter = 0

def plot_confusion_matrix(cm, fig, counter, num_plots, title='Confusion matrix', cmap=plt.cm.Blues):
    ax = fig.add_subplot(1, num_plots, counter+1)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(np.unique(test_labels_df['cat'])))
    plt.set_xticks(tick_marks, np.unique(test_labels_df['cat']), rotation=45)
    plt.set_yticks(tick_marks, np.unique(test_labels_df['cat']))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

fig = plt.figure()
fig.suptitle('Impact of data augmentation using the iris dataset', fontsize=15)
counter = 0
for tree in trees:

    tree_confusion_matrices_mean[tree] = np.zeros(tree_confusion_matrices[tree][0].shape)
    for i in range(len(tree_confusion_matrices[tree])):
        tree_confusion_matrices_mean[tree] = np.add(tree_confusion_matrices_mean[tree], tree_confusion_matrices[tree][i])
    cm_normalized = np.around(tree_confusion_matrices_mean[tree].astype('float') / tree_confusion_matrices_mean[tree].sum(axis=1)[:, np.newaxis], 3)

    diagonal_sum = sum([tree_confusion_matrices_mean[tree][i][i] for i in range(len(tree_confusion_matrices_mean[tree]))])
    total_count = np.sum(tree_confusion_matrices_mean[tree])
    print tree_confusion_matrices_mean[tree], float(diagonal_sum)/float(total_count)

    ax = fig.add_subplot(1, len(trees), counter+1)
    cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_title(titles[counter])
    for (j,i),label in np.ndenumerate(cm_normalized):
        ax.text(i,j,label,ha='center',va='center')
    fig.colorbar(cax,fraction=0.046, pad=0.04)
    counter += 1

F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*1.25, forward=True)
plt.show()