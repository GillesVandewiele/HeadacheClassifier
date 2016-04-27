import os
from pandas.io.parsers import read_csv
from pandas.util.testing import DataFrame

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import StratifiedShuffleSplit

from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor

def plot_learning_curve(title, constructor_names, train_sizes, train_scores_mean, test_scores_mean,
                        train_scores_std, test_scores_std):
    plt.figure()
    plt.title(title)
    plt.ylim([0.0, 1.0])
    plt.xlim([np.min(train_sizes), np.max(train_sizes)])
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    greens = plt.cm.get_cmap('Greens')
    reds = plt.cm.get_cmap('Reds')

    counter = 1
    for name in constructor_names:
        plt.fill_between(train_sizes, np.asarray(train_scores_mean[name]) - np.asarray(train_scores_std[name]),
                         np.asarray(train_scores_mean[name]) + np.asarray(train_scores_std[name]), alpha=0.1,
                         color=reds((1.0/float(len(constructor_names)))*counter))
        plt.fill_between(train_sizes, np.asarray(test_scores_mean[name]) - test_scores_std[name],
                         np.asarray(test_scores_mean[name]) + np.asarray(test_scores_std[name]), alpha=0.1,
                         color=greens((1.0/float(len(constructor_names)))*counter))
        plt.plot(train_sizes, train_scores_mean[name], 'o-', color=reds((1.0/float(len(constructor_names)))*counter),
                 label="Training score "+name, linewidth=1.5)
        plt.plot(train_sizes, test_scores_mean[name], 'o-', color=greens((1.0/float(len(constructor_names)))*counter),
                 label="Test score "+name, linewidth=1.5)
        counter += 1

    plt.legend(loc="best")
    plt.show()

SEED = 1337
N_FOLDS = 5


np.random.seed(SEED)    # 84846513
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join('data', 'heart.dat'), sep=' ')
df.columns=columns

df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)
train_labels_df = labels_df
train_features_df = features_df

c45 = C45Constructor(cf=0.95)
cart = CARTConstructor(max_depth=12, min_samples_leaf=2)
quest = QuestConstructor(default=1, max_nr_nodes=1, discrete_thresh=10, alpha=0.99)

tree_constructors = [c45, cart, quest]

tree_confusion_matrices = {}
for tree_constructor in tree_constructors:
    tree_confusion_matrices[tree_constructor.get_name()] = []

train_sizes = np.linspace(0.1, 1, 10)
mean_train_scores = {}
mean_test_scores = {}
std_train_scores = {}
std_test_scores = {}

for tree_constructor in tree_constructors:
    mean_train_scores[tree_constructor.get_name()] = []
    mean_test_scores[tree_constructor.get_name()] = []
    std_train_scores[tree_constructor.get_name()] = []
    std_test_scores[tree_constructor.get_name()] = []

for size in train_sizes:
    if int(size) == 1:
        sss = StratifiedShuffleSplit(labels_df['cat'], 1, test_size=1-0.99, random_state=SEED)
    else:
        sss = StratifiedShuffleSplit(labels_df['cat'], 1, test_size=1-size, random_state=SEED)
    for train_index, test_index in sss:
        train_features_df, test_features_df = features_df.iloc[train_index, :].copy(), features_df.iloc[test_index, :].copy()
        train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index, :].copy()
        train_features_df = train_features_df.reset_index(drop=True)
        train_labels_df = train_labels_df.reset_index(drop=True)

        skf = sklearn.cross_validation.StratifiedKFold(train_labels_df['cat'], n_folds=N_FOLDS, shuffle=True, random_state=SEED)
        train_scores = {}
        test_scores = {}
        for _train_index, _test_index in skf:
            cv_train_features_df, cv_test_features_df = train_features_df.iloc[_train_index,:].copy(), train_features_df.iloc[_test_index,:].copy()
            cv_train_labels_df, cv_test_labels_df = train_labels_df.iloc[_train_index,:].copy(), train_labels_df.iloc[_test_index,:].copy()
            cv_train_features_df = cv_train_features_df.reset_index(drop=True)
            cv_test_features_df = cv_test_features_df.reset_index(drop=True)
            cv_train_labels_df = cv_train_labels_df.reset_index(drop=True)
            cv_test_labels_df = cv_test_labels_df.reset_index(drop=True)
            cv_train_df = cv_train_features_df.copy()
            cv_train_df['cat'] = cv_train_labels_df['cat'].copy()

            # The decision trees
            for tree_constructor in tree_constructors:
                tree = tree_constructor.construct_tree(cv_train_features_df, cv_train_labels_df)
                tree.populate_samples(cv_train_features_df, cv_train_labels_df['cat'])

                # Calculate train score
                predicted_labels = tree.evaluate_multiple(cv_train_features_df)
                c_mat = tree.plot_confusion_matrix(cv_train_labels_df['cat'].values.astype(str),
                                                   predicted_labels.astype(str))
                train_acc = float(sum([c_mat[i][i] for i in range(len(c_mat))])) / float(c_mat.sum())
                if tree_constructor.get_name() in train_scores:
                    train_scores[tree_constructor.get_name()].append(train_acc)
                else:
                    train_scores[tree_constructor.get_name()] = [train_acc]

                # Calculate test score
                predicted_labels = tree.evaluate_multiple(cv_test_features_df)
                c_mat = tree.plot_confusion_matrix(cv_test_labels_df['cat'].values.astype(str),
                                                   predicted_labels.astype(str))
                test_acc = float(sum([c_mat[i][i] for i in range(len(c_mat))])) / float(c_mat.sum())
                if tree_constructor.get_name() in test_scores:
                    test_scores[tree_constructor.get_name()].append(test_acc)
                else:
                    test_scores[tree_constructor.get_name()] = [test_acc]

        for tree_constructor in tree_constructors:
            mean_train_scores[tree_constructor.get_name()].append(np.mean(train_scores[tree_constructor.get_name()]))
            std_train_scores[tree_constructor.get_name()].append(np.std(train_scores[tree_constructor.get_name()]))
            mean_test_scores[tree_constructor.get_name()].append(np.mean(test_scores[tree_constructor.get_name()]))
            std_test_scores[tree_constructor.get_name()].append(np.std(test_scores[tree_constructor.get_name()]))

print train_sizes
print mean_train_scores
print mean_test_scores

names = []
for tree_constructor in tree_constructors:
    names.append(tree_constructor.get_name())

plot_learning_curve("Learning curve for the heart disease dataset using "+str(N_FOLDS)+"-cross validation", names,
                    np.multiply(train_sizes, len(df)), mean_train_scores, mean_test_scores,
                    std_train_scores, std_test_scores)
