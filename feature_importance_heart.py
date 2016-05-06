import os
from pandas import DataFrame, read_csv
import numpy as np
import operator
import sklearn
import matplotlib.pyplot as plt
from constructors.c45orangeconstructor import C45Constructor

# heart
from extractors.featureselector import RF_feature_selection, boruta_py_feature_selection

SEED = 1337
N_FOLDS = 12

columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join('data', 'heart.dat'), sep=' ')
df.columns = columns

feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['disease']))

for feature in feature_column_names:
    feature_mins[feature] = np.min(df[feature])
    feature_maxs[feature] = np.max(df[feature])
df = df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
train_labels_df = labels_df
train_features_df = features_df

feature_scores = {}

for index, feature in enumerate(feature_column_names):

    c45 = C45Constructor(cf=0.15)

    skf = sklearn.cross_validation.StratifiedKFold(labels_df['cat'], n_folds=N_FOLDS, shuffle=True, random_state=SEED)
    new_features_df = features_df[[feature]]
    acc = 0
    for train_index, test_index in skf:
        train_features_df, test_features_df = new_features_df.iloc[train_index, :].copy(), new_features_df.iloc[test_index, :].copy()
        train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index, :].copy()

        train_features_df = train_features_df.reset_index(drop=True)

        test_features_df = test_features_df.reset_index(drop=True)


        train_labels_df = train_labels_df.reset_index(drop=True)
        test_labels_df = test_labels_df.reset_index(drop=True)

        tree_rf = c45.construct_tree(train_features_df, train_labels_df)

        # tree.visualise(tree_constructor.get_name())
        predicted_labels_rf = tree_rf.evaluate_multiple(test_features_df)

        acc += np.sum([abs(int(a_i) - int(b_i)) for a_i, b_i in zip(predicted_labels_rf, test_labels_df['cat'])])*1.0/len(predicted_labels_rf)
    feature_scores[feature] =acc/N_FOLDS
    print feature
    print acc/N_FOLDS

print "\n\n\n\n------------------------------------------\n\n\n"
sorted_feature_scores = sorted(feature_scores.items(), key=operator.itemgetter(1))
for i in sorted_feature_scores:
    print "%s;%f" % (i[0],i[1])