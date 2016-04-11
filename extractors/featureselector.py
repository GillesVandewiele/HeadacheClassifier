import matplotlib.pyplot as plt
from pandas import DataFrame

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#from util import boruta_py
#from util import boruta_py


def RF_feature_selection(features, labels, features_column_names, plot=False, verbose=False):
    """

    :param features: dataframe of the features
    :param labels: vector containing the correct labels
    :param features_column_names: The column names of the dataframe of the features
    :param plot: Whether to plot the feature importance or not
    :return: vector containing the indices of the most important features (as column number), in order of importance
    """
    rf = RandomForestClassifier(n_estimators=1000, class_weight='auto', n_jobs=-1)
    rf.fit(features, labels)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    feature_importance = []
    if verbose:
        # Print the feature ranking
        print("Feature ranking:")
    for f in range(DataFrame(features).shape[1]):
        if verbose: print("%3d. feature %-25s [%2d] (%9f)" % (
            f + 1, features_column_names[indices[f]], indices[f], importances[indices[f]]))
        feature_importance.append(indices[f])

    if plot:
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(DataFrame(features).shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(DataFrame(features).shape[1]), indices)
        plt.xlim([-1, DataFrame(features).shape[1]])
        plt.show()
    return feature_importance

# def boruta_py_feature_selection(features, labels, column_names, verbose=False):
#     """
#
#     :param features: dataframe of the features
#     :param labels: vector containing the correct labels
#     :param column_names: The column names of the dataframe of the features
#     :param verbose:Whether to print info about the feature importance or not
#     :return: vector containing the indices of the most important features (as column number)
#     """
#     rf = RandomForestClassifier(n_jobs=-1, class_weight='auto')
#     feat_selector = boruta_py.BorutaPy(rf, n_estimators='auto', verbose=0)
#     feat_selector.fit(features, labels)
#     if verbose:
#         print "\n\n\n\n"
#         # check selected features
#         # print feat_selector.support_
#
#         # check ranking of features
#         print "Ranking features: "
#         print feat_selector.ranking_
#
#         # call transform() on X to filter it down to selected features
#         # X_filtered = feat_selector.transform(features)
#         # print X_filtered
#         print "Most important features (%2d):" % sum(feat_selector.support_)
#     important_features = []
#     for i in range(len(feat_selector.support_)):
#         if feat_selector.support_[i]:
#             if verbose: print "feature %2d: %-25s" % (i, column_names[i])
#             important_features.append(i)
#     return important_features