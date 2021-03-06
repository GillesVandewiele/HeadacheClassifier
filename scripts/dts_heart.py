import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import read_csv, DataFrame

import lasagne
import numpy as np
import sklearn
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo
from sklearn.ensemble import RandomForestClassifier

from BN.bayesian_network import learnDiscreteBN, evaluate_multiple
from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from extractors.featureselector import RF_feature_selection#, boruta_py_feature_selection


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

        max_epochs=40000,
        verbose=0,  # set this to 1, if you want to check the val and train scores for each epoch while training.
    )
    return net1

SEED = 1337
N_FOLDS = 5

np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
df.columns=columns

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
train_labels_df = labels_df
train_features_df = features_df

num_features = 8
best_features = RF_feature_selection(features_df.values, labels_df['cat'].tolist(), feature_column_names, verbose=True)
new_features = DataFrame()
for k in range(num_features):
    new_features[feature_column_names[best_features[k]]] = features_df[feature_column_names[best_features[k]]]
features_df = new_features

feature_column_names = list(set(features_df.columns) - set(['disease']))

c45 = C45Constructor(cf=0.01)
cart = CARTConstructor(min_samples_leaf=10, max_depth=6)
quest = QuestConstructor(default=1, max_nr_nodes=1, discrete_thresh=25, alpha=0.05)
tree_constructors = [c45, cart, quest]

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

tree_confusion_matrices = {}
for tree_constructor in tree_constructors:
    tree_confusion_matrices[tree_constructor.get_name()] = []
tree_confusion_matrices["Random Forest"] = []
tree_confusion_matrices["Neural Network"] = []
tree_confusion_matrices["Bayesian Network"] = []
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

    # The decision trees
    for tree_constructor in tree_constructors:
        tree = tree_constructor.construct_tree(train_features_df, train_labels_df)
        tree.populate_samples(train_features_df, train_labels_df['cat'])
        # tree.visualise(tree_constructor.get_name())
        predicted_labels = tree.evaluate_multiple(test_features_df)
        tree_confusion_matrices[tree_constructor.get_name()].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))

    # Random Forest
    rf.fit(train_features_df.values.tolist(), train_labels_df['cat'].tolist())
    predicted_labels = []
    for index, vector in enumerate(test_features_df.values):
        predicted_labels.append(str(rf.predict(vector.reshape(1, -1))[0]))
    tree_confusion_matrices["Random Forest"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels))  # Bit hacky to use the tree method

    train_features_df = (train_features_df - train_features_df.mean()) / (train_features_df.max() - train_features_df.min())
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = (test_features_df - test_features_df.mean()) / (test_features_df.max() - test_features_df.min())
    test_features_df = test_features_df.reset_index(drop=True)

    # Neural Network
    model = build_nn(nr_features=len(train_features_df.columns))
    model.initialize()
    layer_info = PrintLayerInfo()
    layer_info(model)
    y_train = np.reshape(np.asarray(train_labels_df, dtype='int32'), (-1, 1)).ravel()
    model.fit(train_features_df.values, np.add(y_train, -1))
    predicted_labels = []
    for index, vector in enumerate(test_features_df.values):
        predicted_labels.append(str(model.predict(vector.reshape(1, -1))[0]+1))
    tree_confusion_matrices["Neural Network"].append(tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels))  # Bit hacky to use the tree method


    #Bayesian Network
    train_features_df, test_features_df = features_df.iloc[train_index,:].copy(), features_df.iloc[test_index,:].copy()
    train_labels_df, test_labels_df = labels_df.iloc[train_index,:].copy(), labels_df.iloc[test_index,:].copy()
    train_features_df = train_features_df.reset_index(drop=True)
    test_features_df = test_features_df.reset_index(drop=True)
    train_labels_df = train_labels_df.reset_index(drop=True)
    test_labels_df = test_labels_df.reset_index(drop=True)
    train_df = train_features_df.copy()
    train_df['cat'] = train_labels_df['cat'].copy()

    dataframes = learnDiscreteBN(train_df, draw_network=False, continous_columns=feature_column_names,
                                 features_column_names=feature_column_names)
    for i in feature_column_names:
        bins = np.arange((min(test_features_df[i])), (max(test_features_df[i])),
                         ((max(test_features_df[i]) - min(test_features_df[i])) / 5.0))
        test_features_df[i] = pd.pandas.np.digitize(test_features_df[i], bins=bins)

    predicted_labels = evaluate_multiple(test_features_df, dataframes)
    tree_confusion_matrices["Bayesian Network"].append(
        tree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str)))

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