import os
from pandas import read_csv, DataFrame

import numpy as np
import sklearn

from constructors.cartconstructor import CARTConstructor
from extractors.featureselector import RF_feature_selection#, boruta_py_feature_selection

SEED = 1337
N_FOLDS = 5

np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join('data', 'heart.dat'), sep=' ')
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

cart = CARTConstructor(min_samples_leaf=10, max_depth=6)
tree_constructors = [cart]

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
        # tree.populate_samples(train_features_df, train_labels_df['cat'])
        tree.visualise(tree_constructor.get_name())
        predicted_labels = tree.evaluate_multiple(test_features_df)
        s = tree.to_json()
        f = open('cart_tree_heart.json', 'w')
        f.write(s)
        f.close()

