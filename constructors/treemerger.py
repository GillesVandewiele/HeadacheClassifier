import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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
            if region[x_feature][0] == float("-inf"):
                x = 0
            else:
                x = region[x_feature][0]

            if region[x_feature][1] == float("inf"):
                width = x_max - x
            else:
                width = region[x_feature][1] - x

            if region[y_feature][0] == float("-inf"):
                y = 0
            else:
                y = region[y_feature][0]

            if region[y_feature][1] == float("inf"):
                height = y_max - y
            else:
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

    def calculate_intersection(self, regions1, regions2, features):
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