from pandas import read_csv, DataFrame
import numpy as np

import Orange
from sklearn import cross_validation

from constructors.treeconstructor import TreeConstructor
from decisiontree import DecisionTree
from pandas_to_orange import df2table


class C45Constructor(TreeConstructor):

    def __init__(self):
        pass

    def get_name(self):
        return "C4.5"


    def construct_tree(self, training_feature_vectors, labels):

        # First call df2table on the feature table
        orange_feature_table = df2table(training_feature_vectors)

        # Convert classes to strings and call df2table
        labels['cat'] = labels['cat'].apply(str)
        orange_labels_table = df2table(labels)

        # Merge two tables
        orange_table = Orange.data.Table([orange_feature_table, orange_labels_table])

        return self.orange_dt_to_my_dt(Orange.classification.tree.C45Learner(orange_table).tree)

# Evaluate random record
#random_record = orange_feature_table[np.random.choice(len(orange_table), 1)]
#random_record_features_instance = Orange.data.Instance(orange_feature_table.domain, random_record)
#print(c45(random_record_features_instance))


    def orange_dt_to_my_dt(self, orange_dt_root):
        # Check if leaf
        if orange_dt_root.node_type == Orange.classification.tree.C45Node.Leaf:
            return DecisionTree(left=None, right=None, label=(int(orange_dt_root.leaf)+1), data=None, value=None)
        else:
            dt = DecisionTree(label=orange_dt_root.tested.name, data=None, value=orange_dt_root.cut)
            dt.left = self.orange_dt_to_my_dt(orange_dt_root.branch[0])
            dt.right = self.orange_dt_to_my_dt(orange_dt_root.branch[1])
            return dt
"""
# Read csv into pandas frame
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('../heart.dat', sep=' ')
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
df.columns = columns

# Seperate the dataframe into a class dataframe and feature dataframe
labels_df = DataFrame()
labels_df['cat'] = df['disease']
df = df.drop('disease', axis=1)
feature_vectors_df = df.copy()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature_vectors_df, labels_df,test_size=0.25)

c45 = C45Constructor()
decision_tree = c45.construct_tree(feature_vectors_df, labels_df)
decision_tree.populate_samples([1,2], feature_vectors_df, labels_df['cat'].tolist())
decision_tree.visualise('./c45')

predicted_labels = decision_tree.evaluate_multiple(feature_vectors_df)
predicted_labels = [str(int(prediction)+1) for prediction in predicted_labels]
# for barf in range(len(train_labels_df.index)):
#     own_decision_tree.
decision_tree.plot_confusion_matrix(labels_df['cat'], predicted_labels, normalized=True)
"""
