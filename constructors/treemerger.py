import copy
import random
from pandas import DataFrame

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import operator

from objects.decisiontree import DecisionTree
from objects.featuredescriptors import CONTINUOUS


class LineSegment(object):
    """
        Auxiliary class, used for the intersection algorithm
    """
    def __init__(self, lower_bound, upper_bound, region_index):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.region_index = region_index


class DecisionTreeMerger(object):

    def __init__(self):
        pass

    def decision_tree_to_decision_table(self, tree, feature_vectors):
        """
        Convert each path from the root to a leaf into a region, store it into a table
        :param tree: the constructed tree
        :param feature_vectors: the feature vectors of all samples
        :return: a set of regions in a k-dimensional space (k=|feature_vector|), corresponding to the decision tree
        """
        # Initialize an empty region (will be passed on recursively)
        region = {}
        for column in feature_vectors.columns:
            region[column] = [float("-inf"), float("inf")]
            region["class"] = None
        regions = self.tree_to_decision_table(tree, region, [])
        return regions

    def tree_to_decision_table(self, tree, region, regions):
        """
        Recursive method used to convert the decision tree to a decision_table (do not call this one!)
        """
        left_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        right_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        left_region[tree.label][1] = tree.value
        right_region[tree.label][0] = tree.value

        if tree.left.value is None:
            left_region["class"] = tree.left.class_probabilities
            regions.append(left_region)
        else:
            self.tree_to_decision_table(tree.left, left_region, regions)

        if tree.right.value is None:
            right_region["class"] = tree.right.class_probabilities
            regions.append(right_region)
        else:
            self.tree_to_decision_table(tree.right, right_region, regions)

        return regions

    def plot_regions(self, output_path, regions, classes, x_feature, y_feature, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0):
        """
        Given an array of 2dimensional regions (classifying 2 classes), having the following format:
            {x_feature: [lb, ub], y_feature: [lb, ub], 'class': {class_1: prob1, class_2: prob2}}
        We return a rectangle divided in these regions, with a purple color according to the class probabilities
        :param output_path: where does the figure need to be saved
        :param regions: the array of regions, according the format described above
        :param classes: the string representations of the 2 possible classes, this is how they are stored in the
                        "class" dictionary of a region
        :param x_feature: what's the string representation of the x_feature in a region?
        :param y_feature: what's the string representation of the y_feature
        :param x_max: maximum value of x_features
        :param y_max: maximum value of y_features
        :param x_min: minimum value of x_features
        :param y_min: minimum value of y_features
        :return: nothing, but saves a figure to output_path
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.axis([x_min, x_max, y_min, y_max])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        for region in regions:
            x = region[x_feature][0]
            width = region[x_feature][1] - x
            y = region[y_feature][0]
            height = region[y_feature][1] - y

            if classes[0] in region['class'] and classes[1] in region['class']:
                purple_tint = (region['class'][classes[0]], 0.0, region['class'][classes[1]])
            elif classes[0] in region['class']:
                purple_tint = (1.0, 0.0, 0.0)
            elif classes[1] in region['class']:
                purple_tint = (0.0, 0.0, 1.0)
            else:
                print "this shouldn't have happened, go look at treemerger.py"

            ax1.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    width,          # width
                    height,          # height
                    facecolor=purple_tint
                )
            )

        fig1.savefig(output_path)

    def plot_regions_with_points(self, output_path, regions, classes, x_feature, y_feature, points, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0):
        """
        Given an array of 2dimensional regions (classifying 2 classes), having the following format:
            {x_feature: [lb, ub], y_feature: [lb, ub], 'class': {class_1: prob1, class_2: prob2}}
        We return a rectangle divided in these regions, with a purple color according to the class probabilities
        :param output_path: where does the figure need to be saved
        :param regions: the array of regions, according the format described above
        :param classes: the string representations of the 2 possible classes, this is how they are stored in the
                        "class" dictionary of a region
        :param x_feature: what's the string representation of the x_feature in a region?
        :param y_feature: what's the string representation of the y_feature
        :param x_max: maximum value of x_features
        :param y_max: maximum value of y_features
        :param x_min: minimum value of x_features
        :param y_min: minimum value of y_features
        :return: nothing, but saves a figure to output_path
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.axis([x_min, x_max, y_min, y_max])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        for region in regions:
            x = region[x_feature][0]
            width = region[x_feature][1] - x
            y = region[y_feature][0]
            height = region[y_feature][1] - y

            if classes[0] in region['class'] and classes[1] in region['class']:
                purple_tint = (region['class'][classes[0]], 0.0, region['class'][classes[1]])
            elif classes[0] in region['class']:
                purple_tint = (1.0, 0.0, 0.0)
            elif classes[1] in region['class']:
                purple_tint = (0.0, 0.0, 1.0)
            else:
                print "this shouldn't have happened, go look at treemerger.py"

            ax1.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    width,          # width
                    height,          # height
                    facecolor=purple_tint
                )
            )


        for i in range(len(points.index)):
            x = points.iloc[i][x_feature]
            y = points.iloc[i][y_feature]
            ax1.add_patch(
                patches.Circle(
                    (x, y),   # (x,y)
                    0.001,          # width
                    facecolor='black'
                )
            )

        fig1.savefig(output_path)

    def calculate_intersection(self, regions1, regions2, features, feature_maxs, feature_mins):
        """
            Fancy method to calculate intersections. O(n*log(n)) instead of O(n^2)

            Instead of brute force, we iterate over each possible dimension,
            we project each region to that one dimension, creating a line segment. We then construct a set S_i for each
            dimension containing pairs of line segments that intersect in dimension i. In the end, the intersection
            of all these sets results in the intersecting regions. For all these intersection regions, their intersecting
            region is calculated and added to a new set, which is returned in the end
        :param regions1: first set of regions
        :param regions2: second set of regions
        :param features: list of dimension names
        :return: new set of regions, which are the intersections of the regions in 1 and 2
        """
        S_intersections = [None] * len(features)
        for i in range(len(features)):
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

        # The intersection of all these S_i's are the intersecting regions
        intersection_regions_indices = S_intersections[0]
        for k in range(1, len(S_intersections)):
            intersection_regions_indices = self.tuple_list_intersections(intersection_regions_indices, S_intersections[k])

        # Create a new set of regions
        intersected_regions = []
        for intersection_region_pair in intersection_regions_indices:
            region = {}
            for feature in features:
                region[feature] = [max(regions1[intersection_region_pair[0]][feature][0],
                                       regions2[intersection_region_pair[1]][feature][0]),
                                   min(regions1[intersection_region_pair[0]][feature][1],
                                       regions2[intersection_region_pair[1]][feature][1])]
                # Convert all -inf and inf to the mins and max from those features
                if region[feature][0] == float("-inf"):
                    region[feature][0] = feature_mins[feature]
                if region[feature][1] == float("inf"):
                    region[feature][1] = feature_maxs[feature]
            region['class'] = {}
            for key in regions1[intersection_region_pair[0]]['class'].iterkeys():
                region['class'][key] = (regions1[intersection_region_pair[0]]['class'][key] +
                                        regions2[intersection_region_pair[1]]['class'][key]) / 2
            intersected_regions.append(region)

        return intersected_regions

    def tuple_list_intersections(self, list1, list2):
        # Make sure the length of list1 is larger than the length of list2
        if len(list2) > len(list1):
            return self.tuple_list_intersections(list2, list1)
        else:
            intersections = []
            for tuple in list1:
                if tuple in list2 and tuple not in intersections:
                    intersections.append(tuple)

            return intersections

    def calculate_entropy(self, values_list):
        if sum(values_list) == 0:
            return 0
        # Normalize the values by dividing each value by the sum of all values in the list
        normalized_values = map(lambda x: float(x) / float(sum(values_list)), values_list)

        # Calculate the log of the normalized values (negative because these values are in [0,1])

        log_values = map(lambda x: np.log(x)/np.log(2), normalized_values)

        # Take sum of normalized_values * log_values, multiply with (-1) to get positive float
        return -sum(np.multiply(normalized_values, log_values))

    def split_criterion(self, node):
        """
        Calculates information gain ratio (ratio because normal gain tends to have a strong bias in favor of tests with
        many outcomes) for a subtree
        :param node: the node where the information gain needs to be calculated for
        :return: the information gain: information (entropy) in node - sum(weighted information in its children)
        """
        counts_before_split = np.asarray(node.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_count_before_split = sum(counts_before_split)
        info_before_split = self.calculate_entropy(counts_before_split)

        if len(node.left.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            left_counts = np.asarray(node.left.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            left_counts = [0]
        total_left_count = sum(left_counts)
        if len(node.right.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            right_counts = np.asarray(node.right.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            right_counts = [0]
        total_right_count = sum(right_counts)

        # Information gain after split = weighted entropy of left child + weighted entropy of right child
        # weight = number of nodes in child / sum of nodes in left and right child
        info_after_split = float(total_left_count) / float(total_count_before_split) * self.calculate_entropy(left_counts) \
                           + float(total_right_count) / float(total_count_before_split) * self.calculate_entropy(right_counts)

        return (info_before_split - info_after_split) / info_before_split

    def divide_data(self, data, feature, value):
            """
            Divide the data in two subsets, thanks pandas
            :param data: the dataframe to divide
            :param feature: on which column of the dataframe are we splitting?
            :param value: what threshold do we use to split
            :return: node: initialised decision tree object
            """
            return DecisionTree(left=DecisionTree(data=data[data[feature] <= value]),
                                right=DecisionTree(data=data[data[feature] > value]),
                                label=feature,
                                data=data,
                                value=value)

    def dec(self, input_, output_):
        if type(input_) is list:
            for subitem in input_:
                self.dec(subitem, output_)
        else:
            output_.append(input_)

    def generate_samples(self, regions, features, feature_descriptors):
        columns = list(features)
        columns.append('cat')
        samples = DataFrame(columns=columns)
        print "Generating samples for ", len(regions), " regions"
        counter = 0
        for region in regions:
            counter += 1
            region_copy = region.copy()
            del region_copy['class']
            sides = region_copy.items()

            sides = sorted(sides, key=lambda x: x[1][1] - x[1][0])

            amount_of_samples = 1
            for side in sides[:int(math.ceil(np.sqrt(len(features))))]:
                if side[1][1] - side[1][0] > 0:
                    amount_of_samples += (side[1][1] - side[1][0]) * 50

            amount_of_samples *= max(region['class'].iteritems(), key=operator.itemgetter(1))[1]

            print "----> Region ", counter, ": ", int(amount_of_samples), " samples"

            point = {}
            for feature_index in range(len(features)):
                if feature_descriptors[feature_index][0] == CONTINUOUS:
                    point[features[feature_index]] = region[features[feature_index]][0] + \
                                                      ((region[features[feature_index]][1] - region[features[feature_index]][0]) / 2)

            for k in range(int(amount_of_samples)):
                for feature_index in range(len(features)):
                    if feature_descriptors[feature_index][0] == CONTINUOUS:
                        if region[features[feature_index]][1] - region[features[feature_index]][0] > 1.0:
                            point[features[feature_index]] += (random.random() - 0.5) * \
                                                              np.sqrt((region[features[feature_index]][1] - region[features[feature_index]][0]))
                        else:
                            point[features[feature_index]] += (random.random() - 0.5) * \
                                pow((region[features[feature_index]][1] - region[features[feature_index]][0]), 2)
                    else:
                        choice_list = np.arange(region[features[feature_index]][0], region[features[feature_index]][1],
                                                1.0/float(feature_descriptors[feature_index][1])).tolist()
                        if len(choice_list) > 0:
                            choice_list.extend([region[features[feature_index]][0] + (region[features[feature_index]][1] -
                                                                                      region[features[feature_index]][0])/2]*len(choice_list)*2)
                            point[features[feature_index]] = random.choice(choice_list)
                        else:
                            point[features[feature_index]] = region[features[feature_index]][0]

                point['cat'] = max(region['class'].iteritems(), key=operator.itemgetter(1))[0]
                samples = samples.append(point, ignore_index=True)

        return samples

    def regions_to_tree(self, features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=3):
        # TODO: this method is really ugly and inefficient! Needs improvement!!
        # Initialize the feature bounds on their mins and maxs
        bounds = {}
        for _feature in features:
            bounds[_feature] = [feature_mins[_feature], feature_maxs[_feature]]

        # For each feature f, we look for a line that for each f' != f goes from bounds[f'][0] to bounds[f'][1]
        lines = {}
        for _feature in features:
            print "checking for lines ", _feature
            connected_lower_upper_regions = {}
            for other_feature in features:  # Complexity O(d^2) already
                if _feature != other_feature:
                    # Check if we can find 2 points, where the _feature values are the same, and other_feature values
                    # are equal to their required lower and upper bound. Then check if we can draw a line between
                    # these two points. We always check the left line of the region, except for the most left regions

                    # First find all lower regions: their lower bound is equal to the required lower bound (saved in bounds)
                    # and it cannot be the most left line (this is the case when the lower bounds for _feature is equal
                    # to its minimum
                    lower_regions = []
                    for region in regions:  # Complexity O(d^2 * |B|)
                        if region[other_feature][0] == bounds[other_feature][0]:
                            lower_regions.append(region)

                    # Now find upper regions with the same value for _feature as a region in lower_regions
                    lower_upper_regions = []
                    for region in regions:
                        if region[other_feature][1] == bounds[other_feature][1]:
                            for lower_region in lower_regions:  # Even a little bit more complexity here
                                if lower_region[_feature][0] == region[_feature][0] \
                                        and lower_region[_feature][1] == region[_feature][1]:
                                    lower_upper_regions.append([lower_region, region])

                    # for lower_upper_region in lower_upper_regions:
                    #     print lower_upper_region

                    # Now check if we can draw a line between the lower and upper region
                    connected_lower_upper_regions[other_feature] = []
                    for lower_upper_region in lower_upper_regions:
                        lowest_upper_bound = lower_upper_region[0][other_feature][1]
                        highest_lower_bound = lower_upper_region[1][other_feature][0]
                        still_searching = True
                        while still_searching:
                            # Line is connected when either the highest lower bound and lowest upper bound
                            # are adjacent (number of regions along the line is even). Or when
                            # both the lower bounds are equal (odd case, handled further)
                            if highest_lower_bound == lowest_upper_bound:
                                connected_lower_upper_regions[other_feature].append(lower_upper_region)
                                break

                            found_new_lower_bound = False
                            found_new_upper_bound = False
                            for region in regions:  # O boy, this complexity is getting out of hand
                                if found_new_upper_bound and found_new_lower_bound:
                                    break

                                if region[_feature][0] == lower_upper_region[0][_feature][0] \
                                    and region[other_feature][0] == lowest_upper_bound:
                                    found_new_upper_bound = True
                                    lowest_upper_bound = region[other_feature][1]

                                if region[_feature][0] == lower_upper_region[1][_feature][0] \
                                    and region[other_feature][1] == highest_lower_bound:
                                    found_new_lower_bound = True
                                    if region[other_feature][1] == lowest_upper_bound:  # This is the odd case
                                        connected_lower_upper_regions[other_feature].append(lower_upper_region)
                                    highest_lower_bound = region[other_feature][0]

                            still_searching = found_new_lower_bound and found_new_upper_bound

            # Now for all these connected_lower_upper_regions in each dimension, we need to find all line
            # where the value of f is equal (the bounds constraint are already fulfilled)
            if sum([1 if len(value) > 0 else 0 for value in connected_lower_upper_regions.values()]) >= len(features)/2:
                # We found a line fulfilling bounds constraints in all other dimensions
                lines[_feature] = []
                temp = []
                self.dec(connected_lower_upper_regions.values(), temp)
                for region in temp:
                    if region[_feature][0] not in lines[_feature]:
                        lines[_feature].append(region[_feature][0])
            a = 5
        print lines

        # When we looped over each possible feature and found each possible split line, we split the data
        # Using the feature and value of the line and pick the best one
        data = DataFrame(features_df)
        data['cat'] = labels_df
        info_gains = {}
        for key in lines:
            for value in lines[key]:
                node = self.divide_data(data, key, value)
                split_crit = self.split_criterion(node)
                if split_crit > 0:
                    info_gains[node] = self.split_criterion(node)

        print info_gains

        if len(info_gains) > 0:
            best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
            node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)
        else:
            node = DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), data=data, value=None)
            return node
        print node.label, node.value

        ##########################################################################

        # We call recursively with the splitted data and set the bounds of feature f
        # for left child: set upper bounds to the value of the chosen line
        # for right child: set lower bounds to value of chosen line
        feature_mins_right = feature_mins.copy()
        feature_mins_right[node.label] = node.value
        feature_maxs_left = feature_maxs.copy()
        feature_maxs_left[node.label] = node.value
        if len(best_split_node.left.data) >= max_samples and len(best_split_node.right.data) >= max_samples:
            node.left = self.regions_to_tree(best_split_node.left.data.drop('cat', axis=1), best_split_node.left.data[['cat']],
                                        regions, features, feature_mins, feature_maxs_left)
            node.right = self.regions_to_tree(best_split_node.right.data.drop('cat', axis=1), best_split_node.right.data[['cat']],
                                        regions, features, feature_mins_right, feature_maxs)
        else:
            node.label = str(np.argmax(np.bincount(labels_df['cat'].values.astype(int))))
            node.value = None

        return node

    # TODO: write algorithm that returns all possible lines in regions


    # def regions_to_tree_improved(features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=1):
    # # First we create a pandas dataframe with all left-upper points and their widths for each dimension of each region
    # points_columns = []
    # for _feature in features:
    #     points_columns.append(_feature)
    #     points_columns.append(_feature+'_width')
    # points = DataFrame(columns=points_columns)
    # for region in regions:
    #     entry = {}
    #     for _feature in features:
    #         entry[_feature] = region[_feature][0]
    #         entry[_feature+'_width'] = region[_feature][1] - region[_feature][0]
    #     points = points.append(entry, ignore_index=True)
    #
    # print points
    #
    # lines_columns = []
    # for _feature in features:
    #     lines_columns.append(_feature+'_1')
    #     lines_columns.append(_feature+'_2')
    # lines = DataFrame(columns=lines_columns)
    #
    # # For each point and each feature, if point+feature_widths is in the points set as well, there is a line
    # # running there, we try to draw the line as far as possible and add it to a new dataframe
    # for point_index in range(len(points.index)):
    #     for _feature in features:  # try drawing a line across this dimension
    #         current_point = point_index
    #         other_features = list(set(features) - set([_feature]))
    #         searching = True
    #         while searching:
    #             boolean_series = (points[_feature] == (points.iloc[current_point][_feature] +
    #                                                    points.iloc[current_point][_feature+'_width'])).values
    #             boolean_series &= (points[_feature + '_width'] != 0).values
    #
    #             for other_feature in other_features:
    #                 if sum(boolean_series) == 0:
    #                     searching = False
    #                     break
    #                 boolean_series &= (points[other_feature] == points.iloc[current_point][other_feature]).values
    #                 boolean_series &= (points[other_feature+'_width'] != 0).values
    #
    #             entry = {}
    #             for ___feature in features:
    #                 entry[___feature+'_1'] = points.iloc[point_index][___feature]
    #                 entry[___feature+'_2'] = points.iloc[current_point][___feature] + \
    #                                          points.iloc[current_point][___feature+'_width']
    #             lines = lines.append(entry, ignore_index=True)
    #
    #             found_line = np.where(boolean_series == True)
    #
    #             if len(found_line[0]) > 0:
    #                 current_point = found_line[0][0]
    #             else:
    #                 searching = False
    #
    # print lines
    #
    # split_lines = {}
    # for _feature in features:  # try drawing a line across this dimension
    #     split_lines[_feature] = []
    #     other_features = list(set(features) - set([_feature]))
    #     boolean_series = (lines[other_features[0]+'_1'] == feature_mins[other_features[0]])
    #     boolean_series &= (lines[other_features[0]+'_2'] == feature_maxs[other_features[0]])
    #     print np.where(boolean_series==True)
    #     for k in range(1, len(other_features)):
    #         boolean_series &= (lines[other_features[k]+'_1'] == feature_mins[other_features[k]])
    #         boolean_series &= (lines[other_features[k]+'_2'] == feature_maxs[other_features[k]])
    #         print np.where(boolean_series==True)
    #     if sum(boolean_series) > 0:
    #         for index in np.where(boolean_series==True)[0]:
    #             split_lines[_feature].append(lines.iloc[index][_feature+'_1'])
    # for key in split_lines:
    #     split_lines[key] = np.unique(split_lines[key])
    #
    # print split_lines
    #
    # # When we looped over each possible feature and found each possible split line, we split the data
    # # Using the feature and value of the line and pick the best one
    # data = DataFrame(features_df)
    # data['cat'] = labels_df
    # info_gains = {}
    # for key in split_lines:
    #     for value in split_lines[key]:
    #         node = divide_data(data, key, value)
    #         split_crit = split_criterion(node)
    #         if split_crit > 0:
    #             info_gains[node] = split_criterion(node)
    #
    # if len(info_gains) > 0:
    #     best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
    #     node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)
    # else:
    #     node = DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), data=data, value=None)
    #     return node
    #
    # ##########################################################################
    #
    # # We call recursively with the splitted data and set the bounds of feature f
    # # for left child: set upper bounds to the value of the chosen line
    # # for right child: set lower bounds to value of chosen line
    # feature_mins_right = feature_mins.copy()
    # feature_mins_right[node.label] = node.value
    # feature_maxs_left = feature_maxs.copy()
    # feature_maxs_left[node.label] = node.value
    # if len(best_split_node.left.data) >= max_samples and len(best_split_node.right.data) >= max_samples:
    #     node.left = regions_to_tree_improved(best_split_node.left.data.drop('cat', axis=1), best_split_node.left.data[['cat']],
    #                                 regions, features, feature_mins, feature_maxs_left)
    #     node.right = regions_to_tree_improved(best_split_node.right.data.drop('cat', axis=1), best_split_node.right.data[['cat']],
    #                                 regions, features, feature_mins_right, feature_maxs)
    # else:
    #     node.label = str(np.argmax(np.bincount(labels_df['cat'].values.astype(int))))
    #     node.value = None
    #
    # return node