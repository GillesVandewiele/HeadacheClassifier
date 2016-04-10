from pandas import read_csv, DataFrame

import operator
import os

import re
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo

from mpl_toolkits.axes_grid1 import make_axes_locatable

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.featuredescriptors import DISCRETE, CONTINUOUS

def build_nn(nr_features):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],

        input_shape=(None, nr_features),
        hidden_num_units=100,
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        output_num_units=2,
        regression=False,
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=10000,
        verbose=1,  # set this to 1, if you want to check the val and train scores for each epoch while training.
    )
    return net1

SEED = 1337
N_FOLDS = 2

np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join('data', 'heart.dat'), sep=' ')
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

c45 = C45Constructor(cf=0.15)

tree_confusion_matrices = {}
titles = ["Unaugmented C4.5", "Augmented C4.5", "Unaugmented NN", "Augmented NN"]
for title in titles:
    tree_confusion_matrices[title] = []

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

    predicted_labels = tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Unaugmented C4.5"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))
    predicted_labels = augmented_tree.evaluate_multiple(test_features_df)
    tree_confusion_matrices["Augmented C4.5"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))

    train_features_df = (train_features_df - train_features_df.mean()) / (train_features_df.max() - train_features_df.min())
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = (test_features_df - test_features_df.mean()) / (test_features_df.max() - test_features_df.min())
    test_features_df = test_features_df.reset_index(drop=True)

    model = build_nn(nr_features=len(train_features_df.columns))
    model.initialize()
    layer_info = PrintLayerInfo()
    layer_info(model)
    y_train = np.reshape(np.asarray(train_labels_df, dtype='int32'), (-1, 1)).ravel()
    model.fit(train_features_df.values, np.add(y_train, -1))
    predicted_labels = []
    for index, vector in enumerate(test_features_df.values):
        predicted_labels.append(str(model.predict(vector.reshape(1, -1))[0]+1))
    tree_confusion_matrices["Unaugmented NN"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels))  # Bit hacky to use the tree method

    augmented_features_df = (augmented_features_df - augmented_features_df.mean()) / (augmented_features_df.max() - augmented_features_df.min())
    augmented_features_df = augmented_features_df.reset_index(drop=True)

    model = build_nn(nr_features=len(augmented_features_df.columns))
    model.initialize()
    layer_info = PrintLayerInfo()
    layer_info(model)
    y_train = np.reshape(np.asarray(augmented_labels_df, dtype='int32'), (-1, 1)).ravel()
    model.fit(augmented_features_df.values, np.add(y_train, -1))
    predicted_labels = []
    for index, vector in enumerate(test_features_df.values):
        predicted_labels.append(str(model.predict(vector.reshape(1, -1))[0]+1))
    tree_confusion_matrices["Augmented NN"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels))  # Bit hacky to use the tree method


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
fig.suptitle('Impact of data augmentation on heart disease dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
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
F.set_size_inches(Size[0]*2, Size[1]*1.25, forward=True)
plt.show()