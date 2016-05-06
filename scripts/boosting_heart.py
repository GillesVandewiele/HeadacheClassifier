from pandas import read_csv, DataFrame

import os

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.treemerger import DecisionTreeMerger
from objects.featuredescriptors import DISCRETE, CONTINUOUS

SEED = 1337
N_FOLDS = 5
N_BOOSTS = 3

np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
df.columns=columns
descriptors = [(DISCRETE, len(np.unique(df['age']))), (DISCRETE, len(np.unique(df['sex']))),
               (DISCRETE, len(np.unique(df['chest pain type']))), (CONTINUOUS,), (CONTINUOUS,),
               (DISCRETE, len(np.unique(df['fasting blood sugar']))),
               (DISCRETE, len(np.unique(df['resting electrocardio']))), (CONTINUOUS,),
               (DISCRETE, len(np.unique(df['exercise induced angina']))), (CONTINUOUS,),
               (DISCRETE, len(np.unique(df['slope peak']))), (DISCRETE, len(np.unique(df['number of vessels']))),
               (DISCRETE, len(np.unique(df['thal']))), (DISCRETE, len(np.unique(df['disease'])))]

feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['disease']))

for feature in feature_column_names:
        feature_mins[feature] = np.min(df[feature])
        feature_maxs[feature] = np.max(df[feature])
df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
features_df = features_df/features_df.max()
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=0.01)
cart = CARTConstructor(min_samples_leaf=10, max_depth=6)
quest = QuestConstructor(default=1, max_nr_nodes=1, discrete_thresh=25, alpha=0.05)
tree_constructors = [c45, cart, quest]

tree_confusion_matrices = {}
titles = ["C4.5", "Boosted C4.5", "Genetic"]
for title in titles:
    tree_confusion_matrices[title] = []

skf = sklearn.cross_validation.StratifiedKFold(labels_df['cat'], n_folds=N_FOLDS, shuffle=True, random_state=SEED)

for train_index, test_index in skf:
    trees = []
    train_features_df, test_features_df = features_df.iloc[train_index,:].copy(), features_df.iloc[test_index, :].copy()
    train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index, :].copy()
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = test_features_df.reset_index(drop=True)
    train_labels_df = train_labels_df.reset_index(drop=True)
    test_labels_df = test_labels_df.reset_index(drop=True)
    train_df = train_features_df.copy()
    train_df['cat'] = train_labels_df['cat'].copy()

    tree = c45.construct_tree(train_features_df, train_labels_df)
    tree.populate_samples(train_features_df, train_labels_df['cat'])

    predicted_labels = tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["C4.5"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                                                      predicted_labels.astype(str)))
    trees.append(tree)
    for i in range(N_BOOSTS-1):
        missclassified_features = []
        missclassified_labels = []
        for i in range(len(train_features_df)):
            predicted_label = tree.evaluate(train_features_df.iloc[i, :])
            real_label = train_labels_df.iloc[i, :]['cat']
            if real_label != predicted_label:
                missclassified_features.append(train_features_df.iloc[i, :])
                missclassified_labels.append(train_labels_df.iloc[i, :])

        train_features_df = pd.concat([DataFrame(missclassified_features), train_features_df])
        train_labels_df = pd.concat([DataFrame(missclassified_labels), train_labels_df])
        train_features_df = train_features_df.reset_index(drop=True)
        train_labels_df = train_labels_df.reset_index(drop=True)

        tree = c45.construct_tree(train_features_df, train_labels_df)
        tree.populate_samples(train_features_df, train_labels_df['cat'])
        trees.append(tree)

    predicted_labels_set = []
    for tree in trees:
        predicted_labels_set.append(tree.evaluate_multiple(test_features_df))

    predicted_labels = []
    for i in range(len(predicted_labels_set[0])):
        labels = []
        for j in range(len(predicted_labels_set)):
            labels.append(predicted_labels_set[j][i])
        predicted_labels.append(np.argmax(np.bincount(labels)))

    predicted_labels = np.asarray(predicted_labels)
    tree_confusion_matrices["Boosted C4.5"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                                   predicted_labels.astype(str)))

    merger = DecisionTreeMerger()
    best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=10,
                                         num_mutations=5, population_size=10, max_samples=8, val_fraction=0.2,
                                         num_boosts=3)
    predicted_labels = best_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Genetic"].append(best_tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                              predicted_labels.astype(str)))

print tree_confusion_matrices

tree_confusion_matrices_mean = {}

fig = plt.figure()
fig.suptitle('Accuracy on heart disease dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
counter = 0
for key in tree_confusion_matrices:
    tree_confusion_matrices_mean[key] = np.zeros(tree_confusion_matrices[key][0].shape)
    for i in range(len(tree_confusion_matrices[key])):
        tree_confusion_matrices_mean[key] = np.add(tree_confusion_matrices_mean[key], tree_confusion_matrices[key][i])
    cm_normalized = np.around(tree_confusion_matrices_mean[key].astype('float') / tree_confusion_matrices_mean[key].sum(axis=1)[:, np.newaxis], 3)

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

