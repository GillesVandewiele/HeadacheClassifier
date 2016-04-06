# Read csv into pandas frame
import os
from pandas import read_csv, DataFrame

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from extractors.featureselector import RF_feature_selection, boruta_py_feature_selection
from util.metrics import plot_confusion_matrix


def evaluate_random_forests(features_df, labels_df, n_folds=2):
    kf = sklearn.cross_validation.KFold(len(labels_df.index), n_folds=n_folds)

    confusion_matrices_folds = []

    for train, test in kf:
        X_train = DataFrame(features_df, index=train)
        X_test = DataFrame(features_df, index=test)
        y_train = DataFrame(labels_df, index=train)
        y_test = DataFrame(labels_df, index=test)

        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        rf.fit(X_train.values.tolist(), y_train['cat'].tolist())

        predicted_labels = []
        for index, vector in enumerate(X_test.values):
            predicted_labels.append(rf.predict(vector.reshape(1, -1))[0])

        y_test_np = y_test['cat']
        predicted_labels_np = np.array(predicted_labels)
        y_test_np = np.array(y_test_np)
        equal_positions = 1.0 * np.sum(predicted_labels_np == y_test_np)
        accuracy = equal_positions / len(predicted_labels)
        print accuracy

        # Save the confusion matrix for this fold and plot it
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test_np, predicted_labels)
        confusion_matrices_folds.append(confusion_matrix)
        # plt.figure()
        # cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    #     Let's plot the confusion matrix of the avarage confusion matrix
    sum = confusion_matrices_folds[0] * 1.0
    for i in range(1, len(confusion_matrices_folds)):
        sum += confusion_matrices_folds[i]
    sum /= len(confusion_matrices_folds)
    plot_confusion_matrix(sum)





# # TODO: De run code moet hier weg
# columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#            'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
#            'number of vessels', 'thal', 'disease']
#
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
# #Not neccesary to normalize features for RF
# # features_df = (features_df - features_df.mean()) / (features_df.max() - features_df.min())
# features_column_names = features_df.columns
#
# print RF_feature_selection(features_df.values, labels_df['cat'], features_column_names=features_column_names,verbose=True)
# print "\n\n\n\n\n\n\n------------------------------------------------\n\n\n\n\n\n\n"
# print boruta_py_feature_selection(features_df.values, labels_df['cat'].tolist(), column_names=column_names)
# print "\n\n\n\n\n\n\n------------------------------------------------\n\n\n\n\n\n\n"

# Read csv into pandas frame
columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'titanic_train.csv'), sep=',')
df.columns = columns


useful_df = df[['Survived', 'Pclass','Sex','Age','Parch','Fare','Embarked']]
# useful_df = useful_df.dropna()

train_features_df = useful_df[['Pclass','Sex','Age','Parch','Fare','Embarked']].copy()
train_labels_df = useful_df[['Survived']].copy()
train_labels_df.columns = ['cat']
train_labels_df = train_labels_df.reset_index(drop=True)

mapping_sex = {'male': 1, 'female': 2}
mapping_embarked = {'C': 1, 'Q': 2, 'S': 3}
train_features_df['Sex'] = train_features_df['Sex'].map(mapping_sex)
train_features_df['Embarked'] = train_features_df['Embarked'].map(mapping_embarked)

# train_features_df = train_features_df/train_features_df.max()
train_features_df = train_features_df.reset_index(drop=True)

age_avg = train_features_df['Age'].mean()
train_features_df = train_features_df.fillna(age_avg)

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf.fit(train_features_df.values.tolist(), train_labels_df['cat'].tolist())
print ("training done")
columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'titanic_test.csv'), sep=',')
df.columns = columns

test_features_df = df[['PassengerId', 'Pclass','Sex','Age','Parch','Fare','Embarked']]
test_features_df['Sex'] = test_features_df['Sex'].map(mapping_sex)
test_features_df['Embarked'] = test_features_df['Embarked'].map(mapping_embarked)
# test_features_df['Ticket'] = test_features_df['Ticket'].map(ticket_map)
# test_features_df[['Pclass','Sex','Age','Parch','Ticket','Fare','Embarked']] = test_features_df[['Pclass','Sex','Age','Parch','Ticket','Fare','Embarked']]/test_features_df[['Pclass','Sex','Age','Parch','Ticket','Fare','Embarked']].max()



test_features_df = test_features_df.fillna(age_avg)

columns = ['PassengerId', 'Survived']
submission_rf = DataFrame(columns=columns)

test_features_df = test_features_df.fillna(0.5)

for index, vector in enumerate(test_features_df.values):
    submission_rf.loc[len(submission_rf)] = [int(vector[0]),int(rf.predict(vector[1:].reshape(1, -1))[0])]

print submission_rf
submission_rf.to_csv('submission_rf',
                     index=False)