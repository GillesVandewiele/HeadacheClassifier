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
from constructors.treemerger import DecisionTreeMerger
from extractors.featureselector import RF_feature_selection#, boruta_py_feature_selection

SEED = 1337

np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
df.columns=columns
df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
train_labels_df = labels_df
train_features_df = features_df[['max heartrate', 'resting blood pressure']]
feature_mins = {}
feature_maxs = {}
feature_column_names = ['max heartrate', 'resting blood pressure']
for feature in feature_column_names:
        feature_mins[feature] = np.min(df[feature])
        feature_maxs[feature] = np.max(df[feature])

merger = DecisionTreeMerger()
cart = CARTConstructor(min_samples_leaf=10, max_depth=3)

tree = cart.construct_tree(train_features_df, train_labels_df)
tree.populate_samples(train_features_df, train_labels_df['cat'])
tree.visualise("2d_tree")

regions = merger.decision_tree_to_decision_table(tree, train_features_df)
print regions
merger.plot_regions("2d_regions", regions, ['1', '2'],
                    "max heartrate", "resting blood pressure", y_max=feature_maxs["resting blood pressure"],
                    x_max=feature_maxs["max heartrate"], y_min=feature_mins["resting blood pressure"],
                    x_min=feature_mins["max heartrate"])