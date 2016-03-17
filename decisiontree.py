import sklearn
from graphviz import Source
import matplotlib.pyplot as plt
import numpy as np


class DecisionTree(object):

    def __init__(self, right=None, left=None, label='', value=None, data=None, pruning_ratio=0):
        """
        Create a node of the decision tree
        :param right: right child, followed when a feature_value > value
        :param left: left child, followed when feature_value <= value
        :param label: string representation of the attribute the node splits on
                      (feature vector must be dict with same strings and values)
        :param value: the value where the node splits on (if None, then we're in a leaf)
        :param data: pandas dataframe containing the data in the subtree
        """
        self.right = right
        self.left = left
        self.label = label
        self.value = value
        self.data = data
        self.pruning_ratio = pruning_ratio

    def visualise(self, output_path, _view=True, with_pruning_ratio=False):
        """
        visualise the tree, calling convert_node_to_dot
        :param output_path: where the file needs to be saved
        :param with_pruning_ratio: if true, the error rate will be printed too
        :param _view: open the pdf after generation or not
        """
        src = Source(self.convert_to_dot(_with_pruning_ratio=with_pruning_ratio))
        src.render(output_path, view=_view)

    def get_number_of_subnodes(self, count=0):
        """
        Private method using in convert_node_to_dot, in order to give the right child of a node the right count
        :param count: intern parameter, don't set it
        :return: the number of subnodes of a specific node, not including himself
        """
        if self.value is None:
            return count
        else:
            return self.left.get_number_of_subnodes(count=count+1) + self.right.get_number_of_subnodes(count=count+1)

    def convert_node_to_dot(self, count=1, _with_pruning_ratio=False):
        """
        Convert node to dot format in order to visualize our tree using graphviz
        :param count: parameter used to give nodes unique names
        :param _with_pruning_ratio: if true, the error rate will be printed too
        :return: intermediate string of the tree in dot format, without preamble (this is no correct dot format yet!)
        """
        ratio_string = ('(' + str(self.pruning_ratio) + ')') if _with_pruning_ratio else ''
        if self.value is None:
            s = 'Node' + str(count) + ' [label="' + str(self.label) + ratio_string + '" shape="box"];\n'
        else:
            s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + ratio_string + '"];\n'
            s += self.left.convert_node_to_dot(count=count+1, _with_pruning_ratio=_with_pruning_ratio)
            s += 'Node' + str(count) + ' -> ' + 'Node' + str(count+1) + ' [label="true"];\n'
            number_of_subnodes = self.left.get_number_of_subnodes()
            s += self.right.convert_node_to_dot(count=count+number_of_subnodes+2, _with_pruning_ratio=_with_pruning_ratio)
            s += 'Node' + str(count) + '->' + 'Node' + str(count+number_of_subnodes+2) + ' [label="false"];\n'

        return s

    def convert_to_dot(self, _with_pruning_ratio=False):
        """
        Wrapper around convert_node_to_dot (need some preamble and close with })
        :param _with_pruning_ratio: if true, the error rate will be printed too
        :return: the tree in correct dot format
        """
        s = 'digraph DT{\n'
        s += 'node[fontname="Arial"];\n'
        s += self.convert_node_to_dot(_with_pruning_ratio=_with_pruning_ratio)
        s += '}'
        return s

    def to_string(self, tab=0):
        if self.value is None:
            print '\t'*tab + '[', self.label, ']'
        else:
            print '\t'*tab + self.label, ' <= ', self.value
            print '\t'*(tab+1)+ 'LEFT:'
            self.left.to_string(tab=tab+1)
            print '\t'*(tab+1)+ 'RIGHT:'
            self.right.to_string(tab=tab+1)
    def evaluate(self, feature_vector):
        """
        Recursive method to evaluate a feature_vector, the feature_vector must be a dict, having the same
        string representations of the attributes as the representations in the tree
        :param feature_vector: the feature_vector to evaluate
        :return: a class label
        """
        if self.value is None:
            return self.label
        else:
            # feature_vector should only contain 1 row
            if feature_vector[self.label] <= self.value:
                return self.left.evaluate(feature_vector)
            else:
                return self.right.evaluate(feature_vector)

    def evaluate_multiple(self, feature_vectors):
        results = []

        for _index, feature_vector in feature_vectors.iterrows():
            results.append(self.evaluate(feature_vector))
        return np.asarray(results)

    def plot_confusion_matrix(self, actual_labels, predicted_labels, normalized=False, plot=False):
        confusion_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
        return confusion_matrix
        #print("Confusion matrix:\n%s" % confusion_matrix)
        if plot:
            confusion_matrix.plot(normalized=normalized)
            plt.show()
