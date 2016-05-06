import os
from pandas import DataFrame, read_csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from constructors.c45orangeconstructor import C45Constructor

# heart
from extractors.featureselector import RF_feature_selection, boruta_py_feature_selection

SEED = 1337
N_FOLDS = 5

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


best_features_rf = RF_feature_selection(features_df.values, labels_df['cat'].tolist(), feature_column_names,
                                        verbose=True)
best_features_boruta = boruta_py_feature_selection(features_df.values, labels_df['cat'].tolist(), feature_column_names,
                                                   verbose=True)


num_features_rf = len(best_features_boruta)
num_features_boruta = len(best_features_boruta)

new_features_rf = DataFrame()
new_features_boruta = DataFrame()

for k in range(num_features_rf):
    new_features_rf[feature_column_names[best_features_rf[k]]] = features_df[feature_column_names[best_features_rf[k]]]
for k in range(num_features_boruta):
    new_features_boruta[feature_column_names[best_features_boruta[k]]] = features_df[
        feature_column_names[best_features_boruta[k]]]

confusion_matrices = {}
confusion_matrices['RF'] = []
confusion_matrices['Boruta'] = []

features_df_rf = new_features_rf
features_df_boruta = new_features_boruta

feature_column_names_rf = list(set(features_df_rf.columns) - set(['cat']))
feature_column_names_boruta = list(set(features_df_boruta.columns) - set(['cat']))
c45 = C45Constructor(cf=0.15)

skf = sklearn.cross_validation.StratifiedKFold(labels_df['cat'], n_folds=N_FOLDS, shuffle=True, random_state=SEED)

for train_index, test_index in skf:
    train_features_df_rf, test_features_df_rf = features_df_rf.iloc[train_index, :].copy(), features_df_rf.iloc[
                                                                                            test_index, :].copy()
    train_features_df_boruta, test_features_df_boruta = features_df_boruta.iloc[train_index,
                                                        :].copy(), features_df_boruta.iloc[test_index, :].copy()
    train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index, :].copy()

    train_features_df_rf = train_features_df_rf.reset_index(drop=True)
    train_features_df_boruta = train_features_df_boruta.reset_index(drop=True)

    test_features_df_rf = test_features_df_rf.reset_index(drop=True)
    test_features_df_boruta = test_features_df_boruta.reset_index(drop=True)

    train_labels_df = train_labels_df.reset_index(drop=True)
    test_labels_df = test_labels_df.reset_index(drop=True)

    tree_rf = c45.construct_tree(train_features_df_rf, train_labels_df)
    tree_boruta = c45.construct_tree(train_features_df_boruta, train_labels_df)

    # tree.visualise(tree_constructor.get_name())
    predicted_labels_rf = tree_rf.evaluate_multiple(test_features_df_rf)
    predicted_labels_boruta = tree_boruta.evaluate_multiple(test_features_df_boruta)

    confusion_matrices['RF'].append(
        tree_rf.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels_rf.astype(str)))
    confusion_matrices['Boruta'].append(tree_boruta.plot_confusion_matrix(test_labels_df['cat'].values.astype(str),
                                                                          predicted_labels_boruta.astype(str)))

print confusion_matrices

tree_confusion_matrices_mean = {}

fig = plt.figure()
fig.suptitle('Accuracy on heart disease dataset using feature selection RF and Boruta with ' + str(N_FOLDS) + ' folds',
             fontsize=16)
counter = 0
for key in confusion_matrices:
    tree_confusion_matrices_mean[key] = np.zeros(confusion_matrices[key][0].shape)
    for i in range(len(confusion_matrices[key])):
        tree_confusion_matrices_mean[key] = np.add(tree_confusion_matrices_mean[key], confusion_matrices[key][i])
    cm_normalized = np.around(
        tree_confusion_matrices_mean[key].astype('float') / tree_confusion_matrices_mean[key].sum(axis=1)[:,
                                                            np.newaxis], 3)

    diagonal_sum = sum([tree_confusion_matrices_mean[key][i][i] for i in range(len(tree_confusion_matrices_mean[key]))])
    total_count = np.sum(tree_confusion_matrices_mean[key])
    print tree_confusion_matrices_mean[key], float(diagonal_sum) / float(total_count)

    ax = fig.add_subplot(1, len(confusion_matrices), counter + 1)
    cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_title(key, y=1)
    for (j, i), label in np.ndenumerate(cm_normalized):
        ax.text(i, j, label, ha='center', va='center')
    # if counter == len(confusion_matrices) - 1:
    #     fig.colorbar(cax, fraction=0.15, pad=0.05)
    counter += 1

F = plt.gcf()
Size = F.get_size_inches()
# F.set_size_inches(Size[0] *2, Size[1], forward=True)
plt.show()
