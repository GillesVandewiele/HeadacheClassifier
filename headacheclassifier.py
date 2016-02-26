from datacollector import DataCollector


class HeadacheClassifier(object):

    def __init__(self, x=5):
        pass

    def extract_features(self):
        pass

    def select_features(self):
        pass

    def construct_tree(self):
        # Get the data (get_data)
        database = DataCollector.load_data_from_db('localhost', 9000, 'CHRONIC')

        # Extract the features

        # Select the features

        # Construct the tree

        # Return it in some format

    def evaluate(self):
        pass

