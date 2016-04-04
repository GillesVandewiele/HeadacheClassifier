import os
from pandas import read_csv, DataFrame

import lasagne
import numpy as np
import sklearn
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo
from sklearn.cross_validation import StratifiedKFold

from extractors.featureselector import RF_feature_selection
from util import metrics

def build_nn(nr_features):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('dropout1', layers.dropout),
            ('hidden2', layers.DenseLayer),
            # ('dropout2', layers.dropout),
            # ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # # layer parameters
        # l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
        #                              input_var=input_var)

        # input_shape=(None, 1, 28, 28),
        input_shape=(None, nr_features),
        hidden_num_units=150,
        hidden2_num_units=60,
        # hidden3_num_units=150,
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        # batch_iterator_train=BatchIterator(batch_size=16),
        # objective_loss_function=lasagne.objectives.squared_error,
        output_num_units=2,
        regression=False,
        # train_split=TrainSplit(eval_size=0.1),
        dropout1_p=0.8,
        # dropout2_p=0.5,
        # optimization method:
        # hidden_non_linearity = lasagne.nonlinearities.softmax,
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=20000,
        verbose=1,  # set this to 1, if you want to check the val and train scores for each epoch while training.
    )
    return net1


def local_test(feature_vectors_df, labels_df, k=2):
    # labeltjes = [None] * labels_df.values.shape[0]
    labeltjes = labels_df.values
    print labeltjes.shape
    labeltjes = labeltjes
    labeltjes -= 1
    labeltjes = labeltjes.ravel().tolist()
    kf = StratifiedKFold(labeltjes, n_folds=k, shuffle=True)
    # kf = StratifiedKFold(len(feature_vectors_df.index), n_folds=k, shuffle=True)
    # kf = KFold(500, n_folds=k, shuffle=True, random_state=1337)
    confusion_matrices_folds = []

    for train, test in kf:
        # Divide the train_images in a training and validation set (using KFold)
        X_train = feature_vectors_df.values[train, :]
        X_test = feature_vectors_df.values[test, :]

        y_train = [labeltjes[i] for i in train]
        y_test = [labeltjes[i] for i in test]

        # Logistic Regression for feature selection, higher C = more features will be deleted

        # Feature selection/reduction
        model = build_nn(nr_features=X_train.shape[1])
        model.initialize()
        layer_info = PrintLayerInfo()
        layer_info(model)

        # Fit our model

        y_train = np.reshape(np.asarray(y_train, dtype='int32'), (-1, 1)).ravel()
        # print y_train
        # print "Train feature vectors shape: " + X_train.shape.__str__()
        # print "Train labels shape:" + len(y_train).__str__()
        #
        # print "X_train as array shape: " + str(X_train.shape)
        # print "y_train as array shape: " + str(np.reshape(np.asarray(y_train), (-1, 1)).shape)

        model.fit(X_train, np.reshape(np.asarray(y_train), (-1, 1)).ravel())

        preds = model.predict(X_test)
        c = []
        [c.append(preds[i]) if preds[i] == y_test[i] else None for i in range(min(len(y_test), len(preds)))]
        # checks = len([i for i, j in zip(preds, np.reshape(np.asarray(y_train), (-1, 1))) if i == j])

        model = None
        del model
        # Save the confusion matrix for this fold and plot it
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, preds)
        confusion_matrices_folds.append(confusion_matrix)

        # print preds.tolist()
        # print "number of ones: " + str(sum(preds))
        # print y_test
        # print c
        print "Accuracy for fold: " + str(
            ((len(c) * 1.0) / (len(y_test) * 1.0))) + "\n\n\n\n\n-----------------------------\n\n\n"
        #     Let's plot the confusion matrix of the avarage confusion matrix
    sum = confusion_matrices_folds[0] * 1.0
    for i in range(1, len(confusion_matrices_folds)):
        sum += confusion_matrices_folds[i]
    sum /= len(confusion_matrices_folds)
    metrics.plot_confusion_matrix(sum)



# columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#            'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#            'number of vessels', 'thal', 'disease']
# df = read_csv('../data/heart.dat', sep=' ')
# # df = df.iloc[np.random.permutation(len(df))]
# # df = df.reset_index(drop=True)
# df.columns = columns
#
# features_column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral',
#                          'fasting blood sugar', \
#                          'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#                          'number of vessels', 'thal']
# # labels_column_names = 'disease'
# column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#                 'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#                 'number of vessels', 'thal', 'disease']
# df = df[column_names]
# # df = df.drop(columns[:3], axis=1)
# # df = df.drop(columns[4:7], axis=1)
# # df = df.drop(columns[8:-1], axis=1)
# labels_df = DataFrame()
# labels_df['cat'] = df['disease'].copy()
# features_df = df.copy()
# features_df = features_df.drop('disease', axis=1)
# features_df = (features_df - features_df.mean()) / (features_df.max() - features_df.min())
# features_column_names = features_df.columns

# Read csv into pandas frame
columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'titanic_train.csv'), sep=',')
df.columns = columns

useful_df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']]
useful_df = useful_df.dropna()

train_features_df = useful_df[['Sex', 'Age', 'Fare']].copy()
train_labels_df = useful_df[['Survived']].copy()
train_labels_df.columns = ['cat']
train_labels_df = train_labels_df.reset_index(drop=True)

mapping_sex = {'male': 1, 'female': 2}
mapping_embarked = {'C': 1, 'Q': 2, 'S': 3}
train_features_df['Sex'] = train_features_df['Sex'].map(mapping_sex)
# train_features_df['Embarked'] = train_features_df['Embarked'].map(mapping_embarked)

# train_features_df = train_features_df/train_features_df.max()
train_features_df = (train_features_df - train_features_df.mean()) / (train_features_df.max() - train_features_df.min())
train_features_df = train_features_df.reset_index(drop=True)
age_avg = train_features_df['Age'].mean()
train_features_df = train_features_df.fillna(age_avg)

columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'titanic_test.csv'), sep=',')
df.columns = columns

test_features_df = df[['PassengerId','Sex', 'Age', 'Fare']]
test_features_df['Sex'] = test_features_df['Sex'].map(mapping_sex)
# test_features_df['Embarked'] = test_features_df['Embarked'].map(mapping_embarked)
# test_features_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']] = test_features_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']]/test_features_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']].max()
test_features_df[['Sex', 'Age', 'Fare']] = (test_features_df[['Sex', 'Age', 'Fare']]-test_features_df[['Sex', 'Age', 'Fare']].mean())/(test_features_df[['Sex', 'Age', 'Fare']].max()-test_features_df[['Sex', 'Age', 'Fare']].min())

labeltjes = train_labels_df.values

# Logistic Regression for feature selection, higher C = more features will be deleted

print RF_feature_selection(train_features_df.values, train_labels_df['cat'], features_column_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'],verbose=True)
# Feature selection/reduction
model = build_nn(nr_features=train_features_df.values.shape[1])
model.initialize()
layer_info = PrintLayerInfo()
layer_info(model)

# Fit our model

y_train = np.reshape(np.asarray(train_labels_df, dtype='int32'), (-1, 1)).ravel()
# print y_train
# print "Train feature vectors shape: " + X_train.shape.__str__()
# print "Train labels shape:" + len(y_train).__str__()
#
# print "X_train as array shape: " + str(X_train.shape)
# print "y_train as array shape: " + str(np.reshape(np.asarray(y_train), (-1, 1)).shape)

model.fit(train_features_df.values, np.reshape(np.asarray(y_train), (-1, 1)).ravel())

# preds = model.predict(test_features_df)

columns = ['PassengerId', 'Survived']
submission_rf = DataFrame(columns=columns)



test_features_df = test_features_df.fillna(age_avg)

for index, vector in enumerate(test_features_df.values):
    submission_rf.loc[len(submission_rf)] = [int(vector[0]),int(model.predict(vector[1:].reshape(1, -1))[0])]

print submission_rf
submission_rf.to_csv('submission_rf',
                     index=False)