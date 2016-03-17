import copy
from pandas import DataFrame, read_csv

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mp
import pylab as pl
from sklearn.naive_bayes import GaussianNB

from constructors.CARTconstructor import CARTconstructor
from constructors.questconstructor import QuestConstructor
from constructors.C45orangeconstructor import C45Constructor


class TreeEvaluator(object):

    def __init__(self):
        pass

    def evaluate_trees(self, data, tree_constructors, n_folds=2):
        kf = sklearn.cross_validation.KFold(len(data.index), n_folds=n_folds)
        tree_confusion_matrices = {}
        labels_df = DataFrame()
        labels_df['cat'] = data['disease'].copy()
        data = data.drop('disease', axis=1)
        feature_vectors_df = data.copy()
        for train, test in kf:
            X_train = DataFrame(feature_vectors_df, index=train)
            X_test = DataFrame(feature_vectors_df, index=test)
            y_train = DataFrame(labels_df, index=train)
            y_test = DataFrame(labels_df, index=test)
            for tree_constructor in tree_constructors:
                tree = tree_constructor.construct_tree(X_train, y_train)
                tree.visualise(tree_constructor.get_name())
                predicted_labels = tree.evaluate_multiple(X_test)
                if tree_constructor not in tree_confusion_matrices:
                    tree_confusion_matrices[tree_constructor] = [tree.plot_confusion_matrix(y_test['cat'].values, predicted_labels)]
                else:
                    tree_confusion_matrices[tree_constructor].append(tree.plot_confusion_matrix(y_test['cat'].values, predicted_labels))

        fig = plt.figure()
        tree_confusion_matrices_mean = {}
        counter = 1
        for tree_constructor in tree_constructors:
            tree_confusion_matrices_mean[tree_constructor] = np.zeros(tree_confusion_matrices[tree_constructor][0].shape)
            for i in range(n_folds):
                tree_confusion_matrices_mean[tree_constructor] = np.add(tree_confusion_matrices_mean[tree_constructor], tree_confusion_matrices[tree_constructor][i])
            tree_confusion_matrices[tree_constructor] = np.divide(tree_confusion_matrices_mean[tree_constructor], len(tree_confusion_matrices[tree_constructor]))
            tree_confusion_matrices[tree_constructor] = np.divide(tree_confusion_matrices_mean[tree_constructor], np.matrix.sum(np.asmatrix(tree_confusion_matrices_mean[tree_constructor]))).round(3)

            ax = fig.add_subplot(len(tree_constructors), 1, counter)
            cax = ax.matshow(tree_confusion_matrices[tree_constructor], cmap=plt.get_cmap('RdYlGn'))
            ax.set_title(tree_constructor.get_name())
            for (j,i),label in np.ndenumerate(tree_confusion_matrices[tree_constructor]):
                ax.text(i,j,label,ha='center',va='center')
            fig.colorbar(cax)
            counter += 1

        pl.show()

# Read csv into pandas frame
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('heart.dat', sep=' ')
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
df.columns = columns
df = df.drop(columns[:3], axis=1)
df = df.drop(columns[4:7], axis=1)
df = df.drop(columns[8:-1], axis=1)
"""
tree_evaluator = TreeEvaluator()
quest = QuestConstructor()
cart = CARTconstructor()
c45 = C45Constructor()
tree_constructors = [quest, cart, c45]
tree_evaluator.evaluate_trees(df, tree_constructors, n_folds=2)
"""

def decision_tree_to_decision_table(tree, feature_vectors):
    # Convert each path from the root to a leaf into a region, store it into a table
    # Initialize an empty  region (will be passed on recursively)
    region = {}
    for column in feature_vectors.columns:
        region[column] = [float("-inf"), float("inf")]
        region["class"] = None
    regions = tree_to_decision_table(tree, region, [])
    return regions

def tree_to_decision_table(tree, region, regions):
    left_region = copy.deepcopy(region)
    right_region = copy.deepcopy(region)
    left_region[tree.label][1] = tree.value
    right_region[tree.label][0] = tree.value

    # Recursive method
    if tree.left.value is None:
        left_region["class"] = tree.left.label
        regions.append(left_region)
    else:
        tree_to_decision_table(tree.left, left_region, regions)

    if tree.right.value is None:
        right_region["class"] = tree.right.label
        regions.append(right_region)
    else:
        tree_to_decision_table(tree.right, right_region, regions)

    return regions


c45 = C45Constructor()
cart = CARTconstructor()
quest = QuestConstructor()
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
df = df.drop('disease', axis=1)
tree_c45 = c45.construct_tree(df, labels_df)
#tree_c45.visualise("c45_2features")
regions_c45 = decision_tree_to_decision_table(tree_c45, df)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.axis([np.min(df['max heartrate']), np.max(df['max heartrate']), np.min(df['resting blood pressure']), np.max(df['resting blood pressure'])])
plt.xlabel('max heartrate')
plt.ylabel('resting blood pressure')
colors = ["", "red", "blue"]
for region in regions_c45:
    if region['max heartrate'][0] == float("-inf"):
        x = 0
    else:
        x = region['max heartrate'][0]

    if region['max heartrate'][1] == float("inf"):
        width = np.max(df['max heartrate']) - x
    else:
        width = region['max heartrate'][1] - x

    if region['resting blood pressure'][0] == float("-inf"):
        y = 0
    else:
        y = region['resting blood pressure'][0]

    if region['resting blood pressure'][1] == float("inf"):
        height = np.max(df['resting blood pressure']) - y
    else:
        height = region['resting blood pressure'][1] - y


    ax1.add_patch(
        patches.Rectangle(
            (x, y),   # (x,y)
            width,          # width
            height,          # height
            facecolor=colors[region["class"]]
        )
    )

fig1.savefig('rect_c45.png')

tree_cart = cart.construct_tree(df, labels_df)
#tree_cart.visualise("cart_2features")
regions_cart = decision_tree_to_decision_table(tree_cart, df)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.axis([np.min(df['max heartrate']), np.max(df['max heartrate']), np.min(df['resting blood pressure']), np.max(df['resting blood pressure'])])
plt.xlabel('max heartrate')
plt.ylabel('resting blood pressure')
colors = ["", "red", "blue"]
for region in regions_cart:
    if region['max heartrate'][0] == float("-inf"):
        x = 0
    else:
        x = region['max heartrate'][0]

    if region['max heartrate'][1] == float("inf"):
        width = np.max(df['max heartrate']) - x
    else:
        width = region['max heartrate'][1] - x

    if region['resting blood pressure'][0] == float("-inf"):
        y = 0
    else:
        y = region['resting blood pressure'][0]

    if region['resting blood pressure'][1] == float("inf"):
        height = np.max(df['resting blood pressure']) - y
    else:
        height = region['resting blood pressure'][1] - y


    ax1.add_patch(
        patches.Rectangle(
            (x, y),   # (x,y)
            width,          # width
            height,          # height
            facecolor=colors[int(region["class"])]
        )
    )

fig1.savefig('rect_cart.png')

tree_quest = quest.construct_tree(df, labels_df)
#tree_quest.visualise("quest_2features")
regions_quest = decision_tree_to_decision_table(tree_quest, df)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.axis([np.min(df['max heartrate']), np.max(df['max heartrate']), np.min(df['resting blood pressure']), np.max(df['resting blood pressure'])])
plt.xlabel('max heartrate')
plt.ylabel('resting blood pressure')
colors = ["", "red", "blue"]
for region in regions_quest:
    if region['max heartrate'][0] == float("-inf"):
        x = 0
    else:
        x = region['max heartrate'][0]

    if region['max heartrate'][1] == float("inf"):
        width = np.max(df['max heartrate']) - x
    else:
        width = region['max heartrate'][1] - x

    if region['resting blood pressure'][0] == float("-inf"):
        y = 0
    else:
        y = region['resting blood pressure'][0]

    if region['resting blood pressure'][1] == float("inf"):
        height = np.max(df['resting blood pressure']) - y
    else:
        height = region['resting blood pressure'][1] - y


    ax1.add_patch(
        patches.Rectangle(
            (x, y),   # (x,y)
            width,          # width
            height,          # height
            facecolor=colors[region["class"]]
        )
    )

fig1.savefig('rect_quest.png')

print regions_quest
print regions_c45

class LineSegment(object):

    def __init__(self, lower_bound, upper_bound, region_index):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.region_index = region_index

def calculate_intersection(regions1, regions2, features):
    S_intersections = [None] * len(features)
    for i in range(len(features)):
        print "------------------" + features[i] + "------------------"
        # Create B1 and B2: 2 arrays of line segments
        box_set1 = []
        for region_index in range(len(regions1)):
            box_set1.append(LineSegment(regions1[region_index][features[i]][0], regions1[region_index][features[i]][1],
                                        region_index))
            box_set2 = []
        for region_index in range(len(regions2)):
            box_set2.append(LineSegment(regions2[region_index][features[i]][0], regions2[region_index][features[i]][1],
                                        region_index))

        # Sort the two boxsets by their lower bound
        box_set1 = sorted(box_set1, key=lambda segment: segment.lower_bound)
        box_set2 = sorted(box_set2, key=lambda segment: segment.lower_bound)

        # Create a list of unique lower bounds, we iterate over these bounds later
        unique_lower_bounds = []
        for j in range(max(len(box_set1), len(box_set2))):
            if j < len(box_set1) and box_set1[j].lower_bound not in unique_lower_bounds:
                unique_lower_bounds.append(box_set1[j].lower_bound)

            if j < len(box_set2) and box_set2[j].lower_bound not in unique_lower_bounds:
                unique_lower_bounds.append(box_set2[j].lower_bound)

        # Sort them
        unique_lower_bounds = sorted(unique_lower_bounds)

        box1_active_set = []
        box2_active_set = []
        intersections = []
        for lower_bound in unique_lower_bounds:
            # Update all active sets, a region is added when it's lower bound is lower than the current one
            # It is removed when its upper bound is higher than the current lower bound
            for j in range(len(box_set1)):
                if box_set1[j].upper_bound <= lower_bound:
                    if box_set1[j] in box1_active_set:
                        box1_active_set.remove(box_set1[j])
                elif box_set1[j].lower_bound <= lower_bound:
                    if box_set1[j] not in box1_active_set:
                        box1_active_set.append(box_set1[j])
                else:
                    break

            for j in range(len(box_set2)):
                if box_set2[j].upper_bound <= lower_bound:
                    if box_set2[j] in box2_active_set:
                        box2_active_set.remove(box_set2[j])
                elif box_set2[j].lower_bound <= lower_bound:
                    if box_set2[j] not in box2_active_set:
                        box2_active_set.append(box_set2[j])
                else:
                    break

            # All regions from the active set of B1 intersect with the regions in the active set of B2
            for segment1 in box1_active_set:
                for segment2 in box2_active_set:
                    intersections.append((segment1.region_index, segment2.region_index))

        S_intersections[i] = intersections

    print S_intersections
    intersection_regions_indices = S_intersections[0]
    for k in range(1, len(S_intersections)):
        intersection_regions_indices = tuple_list_intersections(intersection_regions_indices, S_intersections[k])

    intersected_regions = []
    for intersection_region_pair in intersection_regions_indices:
        region = {}
        for feature in features:
            region[feature] = [max(regions1[intersection_region_pair[0]][feature][0],
                                   regions2[intersection_region_pair[1]][feature][0]),
                               min(regions1[intersection_region_pair[0]][feature][1],
                                   regions2[intersection_region_pair[1]][feature][1])]
        region["classes"] = [regions1[intersection_region_pair[0]]['class'], regions2[intersection_region_pair[1]]['class']]
        intersected_regions.append(region)

    print intersected_regions

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    plt.axis([np.min(df['max heartrate']), np.max(df['max heartrate']), np.min(df['resting blood pressure']), np.max(df['resting blood pressure'])])
    plt.xlabel('max heartrate')
    plt.ylabel('resting blood pressure')
    colors = ["", "red", "blue"]
    for region in intersected_regions:
        if region['max heartrate'][0] == float("-inf"):
            x = 0
        else:
            x = region['max heartrate'][0]

        if region['max heartrate'][1] == float("inf"):
            width = np.max(df['max heartrate']) - x
        else:
            width = region['max heartrate'][1] - x

        if region['resting blood pressure'][0] == float("-inf"):
            y = 0
        else:
            y = region['resting blood pressure'][0]

        if region['resting blood pressure'][1] == float("inf"):
            height = np.max(df['resting blood pressure']) - y
        else:
            height = region['resting blood pressure'][1] - y

        if region['classes'][0] == region['classes'][1]:
            color = colors[region['classes'][0]]
        else:
            color = "purple"


        ax1.add_patch(
            patches.Rectangle(
                (x, y),   # (x,y)
                width,          # width
                height,          # height
                facecolor=color
            )
        )

    fig1.savefig('intersect.png')

def tuple_list_intersections(list1, list2):
    if len(list2) > len(list1):
        tuple_list_intersections(list2, list1)

    intersections = []
    for tuple in list1:
        if tuple in list2 and tuple not in intersections:
            intersections.append(tuple)

    return intersections


calculate_intersection(regions_c45, regions_quest, df.columns)



"""
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
df = df.drop('disease', axis=1)
feature_vectors_df = df.copy()
gnb = GaussianNB()
y_pred = gnb.fit(feature_vectors_df, labels_df['cat'])
"""

