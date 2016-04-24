import ast
import collections
import os
import pprint
from pandas import read_csv, DataFrame, pandas

import numpy as np
from graphviz import Source
from libpgm.graphskeleton import GraphSkeleton
from libpgm.pgmlearner import PGMLearner




# label_column = "cat"

def learnDiscreteBN(df, continous_columns, features_column_names, label_column='cat', draw_network=False):
    features_df = df.copy()
    features_df = features_df.drop(label_column, axis=1)

    labels_df = DataFrame()
    labels_df[label_column] = df[label_column].copy()

    for i in continous_columns:
        bins = np.arange((min(features_df[i])), (max(features_df[i])),
                         ((max(features_df[i]) - min(features_df[i])) / 5.0))
        features_df[i] = pandas.np.digitize(features_df[i], bins=bins)

    data = []
    for index, row in features_df.iterrows():
        dict = {}
        for i in features_column_names:
            dict[i] = row[i]
        dict[label_column] = labels_df[label_column][index]
        data.append(dict)

    print "Init done"
    learner = PGMLearner()

    test = learner.discrete_estimatebn(data=data, pvalparam=0.05, indegree=1)

    # print test.__dict__

    f = open('heart_structure.txt', 'w')
    s = str(test.__dict__)
    f.write(s)
    f.flush()
    f.close()

    print "done learning"
    edges = test.E
    vertices = test.V
    probas = test.Vdata

    # print probas

    dot_string = 'digraph BN{\n'
    dot_string += 'node[fontname="Arial"];\n'

    dataframes = {}

    print "save data"
    for vertice in vertices:
        print "New vertice: " + str(vertice)
        dataframe = DataFrame()

        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(probas[vertice])
        dot_string += vertice.replace(" ", "_") + ' [label="' + vertice + '\n' + '" ]; \n'

        if len(probas[vertice]['parents']) == 0:
            dataframe['Outcome'] = None
            dataframe['Probability'] = None
            vertex_dict = {}
            for index_outcome, outcome in enumerate(probas[vertice]['vals']):
                vertex_dict[str(outcome)] = probas[vertice]["cprob"][index_outcome]

            od = collections.OrderedDict(sorted(vertex_dict.items()))
            # print "Vertice: " + str(vertice)
            # print "%-7s|%-11s" % ("Outcome", "Probability")
            # print "-------------------"
            for k, v in od.iteritems():
                # print "%-7s|%-11s" % (str(k), str(round(v, 3)))
                dataframe.loc[len(dataframe)] = [k, v]
            dataframes[vertice] = dataframe
        else:
            # pp.pprint(probas[vertice])
            dataframe['Outcome'] = None

            vertexen = {}
            for index_outcome, outcome in enumerate(probas[vertice]['vals']):
                temp = []
                for parent_index, parent in enumerate(probas[vertice]["parents"]):
                    # print str([str(float(index_outcome))])
                    temp = probas[vertice]["cprob"]
                    dataframe[parent] = None
                vertexen[str(outcome)] = temp

            dataframe['Probability'] = None
            od = collections.OrderedDict(sorted(vertexen.items()))

            # [str(float(i)) for i in ast.literal_eval(key)]


            # str(v[key][int(float(k))-1])

            # print "Vertice: " + str(vertice) + " with parents: " + str(probas[vertice]['parents'])
            # print "Outcome" + "\t\t" + '\t\t'.join(probas[vertice]['parents']) + "\t\tProbability"
            # print "------------" * len(probas[vertice]['parents']) *3
            # pp.pprint(od.values())

            counter = 0
            # print number_of_cols
            for outcome, cprobs in od.iteritems():
                for key in cprobs.keys():
                    array_frame = []
                    array_frame.append((outcome))
                    print_string = str(outcome) + "\t\t"
                    for parent_value, parent in enumerate([i for i in ast.literal_eval(key)]):
                        # print "parent-value:"+str(parent_value)
                        # print "parten:"+str(parent)
                        array_frame.append(int(float(parent)))
                        # print "lengte array_frame: "+str(len(array_frame))
                        print_string += parent + "\t\t"
                    array_frame.append(cprobs[key][counter])
                    # print "lengte array_frame (2): "+str(len(array_frame))
                    # print  cprobs[key][counter]
                    print_string += str(cprobs[key][counter]) + "\t"
                    # for stront in [str(round(float(i), 3)) for i in ast.literal_eval(key)]:
                    #     print_string += stront + "\t\t"
                    # print "print string: " + print_string
                    # print "array_frame:" + str(array_frame)
                    dataframe.loc[len(dataframe)] = array_frame
                counter += 1
        print "Vertice " + str(vertice) + " done"
        dataframes[vertice] = dataframe

    for edge in edges:
        dot_string += edge[0].replace(" ", "_") + ' -> ' + edge[1].replace(" ", "_") + ';\n'

    dot_string += '}'
    src = Source(dot_string)
    if draw_network:src.render('../data/BN', view=draw_network)
    if draw_network:src.render('../data/BN', view=False)
    print "vizualisation done"
    return dataframes


def learnDiscreteBN_with_structure(df, continous_columns, features_column_names, label_column='cat',
                                   draw_network=False):
    features_df = df.copy()
    features_df = features_df.drop(label_column, axis=1)

    labels_df = DataFrame()
    labels_df[label_column] = df[label_column].copy()

    for i in continous_columns:
        bins = np.arange((min(features_df[i])), (max(features_df[i])),
                         ((max(features_df[i]) - min(features_df[i])) / 5.0))
        features_df[i] = pandas.np.digitize(features_df[i], bins=bins)

    data = []
    for index, row in features_df.iterrows():
        dict = {}
        for i in features_column_names:
            dict[i] = row[i]
        dict[label_column] = labels_df[label_column][index]
        data.append(dict)

    print "Init done"
    learner = PGMLearner()

    graph = GraphSkeleton()

    graph.V = []
    graph.E = []

    graph.V.append(label_column)

    for vertice in features_column_names:
        graph.V.append(vertice)
        graph.E.append([vertice, label_column])

    test = learner.discrete_mle_estimateparams(graphskeleton=graph, data=data)

    print "done learning"

    edges = test.E
    vertices = test.V
    probas = test.Vdata

    # print probas

    dot_string = 'digraph BN{\n'
    dot_string += 'node[fontname="Arial"];\n'

    dataframes = {}

    print "save data"
    for vertice in vertices:
        print "New vertice: " + str(vertice)
        dataframe = DataFrame()

        pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(probas[vertice])
        dot_string += vertice.replace(" ", "_") + ' [label="' + vertice + '\n' + '" ]; \n'

        if len(probas[vertice]['parents']) == 0:
            dataframe['Outcome'] = None
            dataframe['Probability'] = None
            vertex_dict = {}
            for index_outcome, outcome in enumerate(probas[vertice]['vals']):
                vertex_dict[str(outcome)] = probas[vertice]["cprob"][index_outcome]

            od = collections.OrderedDict(sorted(vertex_dict.items()))
            # print "Vertice: " + str(vertice)
            # print "%-7s|%-11s" % ("Outcome", "Probability")
            # print "-------------------"
            for k, v in od.iteritems():
                # print "%-7s|%-11s" % (str(k), str(round(v, 3)))
                dataframe.loc[len(dataframe)] = [k, v]
            dataframes[vertice] = dataframe
        else:
            # pp.pprint(probas[vertice])
            dataframe['Outcome'] = None

            vertexen = {}
            for index_outcome, outcome in enumerate(probas[vertice]['vals']):
                temp = []
                for parent_index, parent in enumerate(probas[vertice]["parents"]):
                    # print str([str(float(index_outcome))])
                    temp = probas[vertice]["cprob"]
                    dataframe[parent] = None
                vertexen[str(outcome)] = temp

            dataframe['Probability'] = None
            od = collections.OrderedDict(sorted(vertexen.items()))

            # [str(float(i)) for i in ast.literal_eval(key)]


            # str(v[key][int(float(k))-1])

            # print "Vertice: " + str(vertice) + " with parents: " + str(probas[vertice]['parents'])
            # print "Outcome" + "\t\t" + '\t\t'.join(probas[vertice]['parents']) + "\t\tProbability"
            # print "------------" * len(probas[vertice]['parents']) *3
            # pp.pprint(od.values())

            counter = 0
            # print number_of_cols
            for outcome, cprobs in od.iteritems():
                for key in cprobs.keys():
                    array_frame = []
                    array_frame.append((outcome))
                    print_string = str(outcome) + "\t\t"
                    for parent_value, parent in enumerate([i for i in ast.literal_eval(key)]):
                        # print "parent-value:"+str(parent_value)
                        # print "parten:"+str(parent)
                        array_frame.append(int(float(parent)))
                        # print "lengte array_frame: "+str(len(array_frame))
                        print_string += parent + "\t\t"
                    array_frame.append(cprobs[key][counter])
                    # print "lengte array_frame (2): "+str(len(array_frame))
                    # print  cprobs[key][counter]
                    print_string += str(cprobs[key][counter]) + "\t"
                    # for stront in [str(round(float(i), 3)) for i in ast.literal_eval(key)]:
                    #     print_string += stront + "\t\t"
                    # print "print string: " + print_string
                    # print "array_frame:" + str(array_frame)
                    dataframe.loc[len(dataframe)] = array_frame
                counter += 1
        print "Vertice " + str(vertice) + " done"
        dataframes[vertice] = dataframe

    for edge in edges:
        dot_string += edge[0].replace(" ", "_") + ' -> ' + edge[1].replace(" ", "_") + ';\n'

    dot_string += '}'
    # src = Source(dot_string)
    # src.render('../data/BN', view=draw_network)
    # src.render('../data/BN', view=False)
    print "vizualisation done"
    return dataframes


def eval_sample(feature_dict, dataframes, label_column='cat', verbose=False):
    to_predict = label_column
    if verbose:
        print "Evaluating the %s for sample with observed features: %s" % (to_predict, str(feature_dict.keys()))
    df = dataframes[to_predict].copy()
    for feature in feature_dict.keys():
        if verbose: print "Set value for feature %s to %s" % (feature, str(feature_dict[feature]))
        if feature in df.columns:
            df = df[df[feature] - int(feature_dict[feature]) == 0]
            df = df.drop(feature, axis=1)
    if verbose: print df
    # return df['Probability'].tolist(), df
    if len(df) == len(df['Outcome']):
        return np.unique(df['Outcome'])[np.argmax(df['Probability'].tolist())]


def evaluate_multiple(feature_dicts, dataframes, label_column='cat'):
    """
    Wrapper method to evaluate multiple vectors at once (just a for loop where evaluate is called)
    :param feature_vectors: the feature_vectors you want to evaluate
    :return: list of class labels
    """
    results = []

    for i in range(len(feature_dicts)):
        feature_dict = feature_dicts.iloc[i, :].to_dict()
        results.append(eval_sample(feature_dict, dataframes, label_column))
    return np.asarray(results)


#########################################################
#                         INIT                          #
#########################################################

# columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#            'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#            'number of vessels', 'thal', 'disease']
#
# continous_columns = ['age', 'resting blood pressure', 'oldpeak', 'max heartrate', 'serum cholestoral',
#                      'max heartrate']
#
# df = read_csv('../data/heart.dat', sep=' ')
# # df = df.iloc[np.random.permutation(len(df))]
# # df = df.reset_index(drop=True)
# df.columns = columns
#
# features_column_names = columns[0:len(columns) - 1]
#
# labels_column_name = 'disease'
# column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral',
#                 'fasting blood sugar', \
#                 'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#                 'number of vessels', 'thal', 'disease']
# df = df[column_names]
# # df = df.drop(columns[:3], axis=1)
# # df = df.drop(columns[4:7], axis=1)
# # df = df.drop(columns[8:-1], axis=1)
# labels_df = DataFrame()
# labels_df[labels_column_name] = df[labels_column_name].copy()
#
# dataframes = learnDiscreteBN(df, draw_network=True, continous_columns=continous_columns,
#                                features_column_names=features_column_names, label_column=labels_column_name)
# # print dataframes['cat']
# dict_features = {}
# dict_features['number of vessels'] = 1
# dict_features['chest pain type'] = 1
# dict_features['thal'] = 3
# # print eval_sample({}, dataframes, label_column='disease', verbose=True)
#
# print eval_sample(dict_features, dataframes, label_column='disease', verbose=True)
