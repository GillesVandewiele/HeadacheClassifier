from pandas import read_csv, DataFrame
import numpy as np

import Orange

from decisiontree import DecisionTree
from pandas_to_orange import df2table

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


# First call df2table on the feature table
orange_feature_table = df2table(feature_vectors_df)

# Convert classes to strings and call df2table
labels_df['cat'] = labels_df['cat'].apply(str)
orange_labels_table = df2table(labels_df)

# Merge two tables
orange_table = Orange.data.Table([orange_feature_table, orange_labels_table])
for d in orange_table[:3]:
    print([d[i].value for i in range(len(d)-1)])

c45 = Orange.classification.tree.C45Learner(orange_table)

# Evaluate random record
random_record = orange_feature_table[np.random.choice(len(orange_table), 1)]
random_record_features_instance = Orange.data.Instance(orange_feature_table.domain, random_record)
print(c45(random_record_features_instance))


def orange_dt_to_my_dt(orange_dt_root):
    # Check if leaf
    if orange_dt_root.node_type == Orange.classification.tree.C45Node.Leaf:
        return DecisionTree(left=None, right=None, label=orange_dt_root.leaf, data=None, value=None)
    else:
        dt = DecisionTree(label=orange_dt_root.tested.name, data=None, value=orange_dt_root.cut)
        dt.left = orange_dt_to_my_dt(orange_dt_root.branch[0])
        dt.right = orange_dt_to_my_dt(orange_dt_root.branch[1])
        return dt

print(c45)
print(c45.tree.leaf)
my_dt = orange_dt_to_my_dt(c45.tree)

#my_dt.visualise("../orange_tree")
