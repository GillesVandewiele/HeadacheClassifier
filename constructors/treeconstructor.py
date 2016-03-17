

class TreeConstructor(object):

    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError("This method needs to be implemented")

    def split_criterion(self, node):
        raise NotImplementedError("This method needs to be implemented")

    def construct_tree(self, training_feature_vectors, labels, default, max_nr_nodes=1):
        raise NotImplementedError("This method needs to be implemented")

    def post_prune(self, tree, testing_feature_vectors, labels):
        raise NotImplementedError("This method needs to be implemented")

