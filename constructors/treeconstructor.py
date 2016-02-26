

class TreeConstructor(object):

    def __init__(self):
        pass

    def split_criterion(self, node):
        raise NotImplementedError("This method needs to be implemented")

    def construct_tree(self, feature_vectors, labels):
        raise NotImplementedError("This method needs to be implemented")

