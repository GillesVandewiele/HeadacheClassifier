from pandas import read_csv, DataFrame, pandas

import numpy as np
from libpgm.pgmlearner import PGMLearner


def learnDiscreteBN():
    columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
               'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
               'number of vessels', 'thal', 'disease']

    continous_columns = ['age', 'resting blood pressure','oldpeak', 'max heartrate', 'serum cholestoral', 'max heartrate']

    df = read_csv('../data/heart.dat', sep=' ')
    # df = df.iloc[np.random.permutation(len(df))]
    # df = df.reset_index(drop=True)
    df.columns = columns

    features_column_names = columns[0:len(columns)-1]

    # labels_column_names = 'disease'
    column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral',
                    'fasting blood sugar', \
                    'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
                    'number of vessels', 'thal', 'disease']
    df = df[column_names]
    # df = df.drop(columns[:3], axis=1)
    # df = df.drop(columns[4:7], axis=1)
    # df = df.drop(columns[8:-1], axis=1)
    labels_df = DataFrame()
    labels_df['cat'] = df['disease'].copy()
    features_df = df.copy()
    features_df = features_df.drop('disease', axis=1)

    for i in continous_columns:
        # features_df[i] = pandas.qcut(features_df[i], 5, labels=False)
        print i
        print "min: " + str(min(features_df[i]))
        print "max: " + str(max(features_df[i]))
        print "step: " + str((max(features_df[i])-min(features_df[i]))/5.0)
        bins = np.arange((min(features_df[i])), (max(features_df[i])), ((max(features_df[i])-min(features_df[i]))/5.0))
        features_df[i] = pandas.np.digitize(features_df[i], bins=bins)

    data = []
    for index, row in features_df.iterrows():
        dict = {}
        for i in features_column_names:
            dict[i] = row[i]
        dict['cat'] = labels_df['cat'][index]
        data.append(dict)

    learner = PGMLearner()

    test = learner.discrete_estimatebn(data=data, pvalparam=0.05, indegree=1)
    print "Test"


learnDiscreteBN()
