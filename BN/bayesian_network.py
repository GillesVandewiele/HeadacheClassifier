import ast
import collections
import pprint
from lxml.html import open_in_browser
from pandas import read_csv, DataFrame, pandas

import numpy as np
from graphviz import Source
from libpgm.pgmlearner import PGMLearner
from lxml.etree import HTML
import lxml
from tabulate import tabulate


def learnDiscreteBN():
    columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
               'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
               'number of vessels', 'thal', 'disease']

    continous_columns = ['age', 'resting blood pressure', 'oldpeak', 'max heartrate', 'serum cholestoral',
                         'max heartrate']

    df = read_csv('../data/heart.dat', sep=' ')
    # df = df.iloc[np.random.permutation(len(df))]
    # df = df.reset_index(drop=True)
    df.columns = columns

    features_column_names = columns[0:len(columns) - 1]

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

    # Read csv into pandas frame
    # columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    # # columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
    # #            'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
    # #            'number of vessels', 'thal', 'disease']
    # # df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
    # df = read_csv(os.path.join(os.path.join('..', 'data'), 'car.data'), sep=',')
    # df.columns = columns
    # mapping_buy_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    # mapping_doors = {'2': 0, '3': 1, '4': 2, '5more': 3}
    # mapping_persons = {'2': 0, '4': 1, 'more': 2}
    # mapping_lug = {'small': 0, 'med': 1, 'big': 2}
    # mapping_safety = {'low': 0, 'med': 1, 'high': 2}
    # mapping_class = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
    # df['maint'] = df['maint'].map(mapping_buy_maint)
    # df['buying'] = df['buying'].map(mapping_buy_maint)
    # df['doors'] = df['doors'].map(mapping_doors)
    # df['persons'] = df['persons'].map(mapping_persons)
    # df['lug_boot'] = df['lug_boot'].map(mapping_lug)
    # df['safety'] = df['safety'].map(mapping_safety)
    # df['class'] = df['class'].map(mapping_class)
    # # permutation = np.random.permutation(df.index)
    # # df = df.reindex(permutation)
    # # df = df.reset_index(drop=True)
    # # df = df.head(300)
    #
    # labels_df = DataFrame()
    # labels_df['cat'] = df['class'].copy()
    # features_df = df.copy()
    # features_df = features_df.drop('class', axis=1)
    # features_column_names = features_df.columns

    for i in continous_columns:
        # features_df[i] = pandas.qcut(features_df[i], 5, labels=False)
        # print i
        # print "min: " + str(min(features_df[i]))
        # print "max: " + str(max(features_df[i]))
        # print "step: " + str((max(features_df[i])-min(features_df[i]))/5.0)
        bins = np.arange((min(features_df[i])), (max(features_df[i])),
                         ((max(features_df[i]) - min(features_df[i])) / 5.0))
        features_df[i] = pandas.np.digitize(features_df[i], bins=bins)

    data = []
    for index, row in features_df.iterrows():
        dict = {}
        for i in features_column_names:
            dict[i] = row[i]
        dict['cat'] = labels_df['cat'][index]
        data.append(dict)

    learner = PGMLearner()

    test = learner.discrete_estimatebn(data=data, pvalparam=0.9, indegree=1)

    edges = test.E
    vertices = test.V
    probas = test.Vdata

    # print probas



    dot_string = 'digraph BN{\n'
    dot_string += 'node[fontname="Arial"];\n'

    dataframes = {}

    for vertice in vertices:
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
                print "%-7s|%-11s" % (str(k), str(round(v, 3)))
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
            array_frame = []
            counter = 0
            len_outcome = len(od.keys())
            for outcome, cprobs in od.iteritems():
                for key in cprobs.keys():
                    array_frame.append((outcome))
                    print_string = str(int(float(outcome))) + "\t\t"
                    for parent_value, parent in enumerate([i for i in ast.literal_eval(key)]):
                        array_frame.append(parent)
                        print_string += parent + "\t\t"
                    array_frame.append(cprobs[key][counter])

                    print_string += str(cprobs[key][counter]) +"\t"
                    # for stront in [str(round(float(i), 3)) for i in ast.literal_eval(key)]:
                    #     print_string += stront + "\t\t"
                    # print print_string
                    dataframe.loc[len(dataframe)] = array_frame

                counter+=1

        dataframes[vertice] = dataframe
        # print tabulate([list(row) for row in dataframe.values], headers=list(dataframe.columns))
        print  "\n\n\n\n\n"
        print dataframe.head(n=100)
        # print (dataframe.to_html()).display()
        # open_in_browser(HTML(dataframe.to_html()))

    for edge in edges:
        dot_string += edge[0].replace(" ", "_") + ' -> ' + edge[1].replace(" ", "_") + ';\n'

    #
    # s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + ratio_string + '"];\n'
    # s += self.left.convert_node_to_dot(count=count + 1, _with_pruning_ratio=_with_pruning_ratio)
    # s += 'Node' + str(count) + ' -> ' + 'Node' + str(count + 1) + ' [label="true"];\n'
    # number_of_subnodes = self.left.get_number_of_subnodes()
    # s += self.right.convert_node_to_dot(count=count + number_of_subnodes + 2,

    dot_string += '}'
    src = Source(dot_string)
    src.render('../data/BN', view=False)


learnDiscreteBN()
