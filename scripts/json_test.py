from pandas import read_csv, DataFrame

import operator
import os

import re
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo

from mpl_toolkits.axes_grid1 import make_axes_locatable

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.decisiontree import DecisionTree
from objects.featuredescriptors import DISCRETE, CONTINUOUS

columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
df.columns=columns


labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()
features_df = df.copy()
features_df = features_df.drop('disease', axis=1)

c45 = C45Constructor(cf=0.15)
tree = c45.construct_tree(features_df, labels_df)
tree.visualise("test")
tree.populate_samples(features_df, labels_df['cat'])
json = tree.to_json()
DecisionTree.from_json(json).visualise("test2")