from pandas import read_csv, DataFrame

import operator
import os

import re
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor
from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from objects.featuredescriptors import DISCRETE, CONTINUOUS

SEED = 13337
N_FOLDS = 5

np.random.seed(SEED)    # 84846513


columns = ['PassengerId','Survived', 'Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join('data', 'titanic_train.csv'), sep=',')
df.columns = columns
df = df[['Pclass', 'Name', 'Sex', 'Ticket', 'Parch', 'Age', 'SibSp', 'Fare', 'Embarked', 'Survived']].copy()
df = df.dropna()

mapping_sex = {'male': 1, 'female': 2}
mapping_embarked = {'C': 1, 'Q': 2, 'S': 3}
df['Sex'] = df['Sex'].map(mapping_sex)
df['Embarked'] = df['Embarked'].map(mapping_embarked)

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

df['Title'] = df['Name'].map(lambda x: get_title(x))
df = df.drop('Name', axis=1)
mapping_title = {}
counter = 1
for value in np.unique(df['Title']):
    mapping_title[value] = counter
    counter += 1
df['Title'] = df['Title'].map(mapping_title)


def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    match = re.compile("([\d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

# extract and massage the ticket prefix
df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )
df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )

# create binary features for each prefix
#prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
#df = pd.concat([df, prefixes], axis=1)

# factorize the prefix to create a numerical categorical variable
df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]

# extract the ticket number
df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )

# create a feature for the number of digits in the ticket number
df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)

# create a feature for the starting number of the ticket number
df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)

# The prefix and (probably) number themselves aren't useful
df.drop(['TicketPrefix', 'TicketNumber', 'Ticket'], axis=1, inplace=True)

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
# df['class'] = df['class'].map(mapping_class).astype(int)
df=df.reset_index(drop=True)
labels_df = DataFrame()
labels_df['cat'] = df['Survived'].copy()
features_df = df.copy()
features_df = features_df.drop('Survived', axis=1)

feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['Survived']))

for feature in feature_column_names:
        feature_mins[feature] = np.min(df[feature])
        feature_maxs[feature] = np.max(df[feature])

c45 = C45Constructor(cf=0.15)
cart = CARTConstructor(min_samples_leaf=20)
quest = QuestConstructor(default=1, max_nr_nodes=10, discrete_thresh=10, alpha=0.15)
tree_constructors = [c45, cart, quest]

merger = DecisionTreeMerger()
train_df = features_df.copy()
train_df['cat'] = labels_df['cat'].copy()
best_tree = merger.genetic_algorithm(train_df, 'cat', tree_constructors, seed=SEED, num_iterations=5,
                                                    num_mutations=3, population_size=7)
c45_tree = c45.construct_tree(features_df, labels_df)
c45_tree.populate_samples(features_df, labels_df['cat'])
best_tree.visualise('best_tree')
c45_tree.visualise('c45')
c45_regions = merger.decision_tree_to_decision_table(c45_tree, features_df)

columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join('data', 'titanic_test.csv'), sep=',')
df.columns = columns

test_features_df = df[['PassengerId', 'Pclass', 'Sex', 'Parch', 'Age', 'Name', 'Ticket', 'SibSp', 'Fare', 'Embarked']].copy()
test_features_df['Sex'] = test_features_df['Sex'].map(mapping_sex)
test_features_df['Embarked'] = test_features_df['Embarked'].map(mapping_embarked)
test_features_df['Title'] = test_features_df['Name'].map(lambda x: get_title(x))
test_features_df = test_features_df.drop('Name', axis=1)
test_features_df['Title'] = test_features_df['Title'].map(mapping_title)

# extract and massage the ticket prefix
test_features_df['TicketPrefix'] = test_features_df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
test_features_df['TicketPrefix'] = test_features_df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )
test_features_df['TicketPrefix'] = test_features_df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )

# create binary features for each prefix
# prefixes = pd.get_dummies(test_features_df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
# test_features_df = pd.concat([test_features_df, prefixes], axis=1)

# factorize the prefix to create a numerical categorical variable
test_features_df['TicketPrefixId'] = pd.factorize(test_features_df['TicketPrefix'])[0]

# extract the ticket number
test_features_df['TicketNumber'] = test_features_df['Ticket'].map( lambda x: getTicketNumber(x) )

# create a feature for the number of digits in the ticket number
test_features_df['TicketNumberDigits'] = test_features_df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)

# create a feature for the starting number of the ticket number
test_features_df['TicketNumberStart'] = test_features_df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)

# The prefix and (probably) number themselves aren't useful
test_features_df.drop(['TicketPrefix', 'TicketNumber', 'Ticket'], axis=1, inplace=True)
test_features_df = test_features_df.reset_index(drop=True)
columns = ['PassengerId', 'Survived']
submission_merge = DataFrame(columns=columns)
submission_c45 = DataFrame(columns=columns)

for i in range(len(test_features_df.index)):
    sample = test_features_df.loc[i]
    submission_merge.loc[len(submission_merge)] = [int(sample['PassengerId']), best_tree.evaluate(sample)]
    submission_c45.loc[len(submission_c45)] = [int(sample['PassengerId']), c45_tree.evaluate(sample)]


submission_merge.to_csv('submission_genetic', index=False)
submission_c45.to_csv('submission_c45', index=False)
