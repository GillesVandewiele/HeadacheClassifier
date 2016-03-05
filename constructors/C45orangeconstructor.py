from pandas import read_csv, DataFrame
import numpy as np

import Orange

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
    print(d)

c45 = Orange.classification.tree.C45Learner(orange_table)
print(c45)
print(c45.__dict__)
