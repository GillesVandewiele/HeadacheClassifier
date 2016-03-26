# Read csv into pandas frame
import numpy as np
import sklearn
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestClassifier

# TODO: nieuwe methode naam please
def evaluate_trees(features_df, labels_df, n_folds=2):
    kf = sklearn.cross_validation.KFold(len(labels_df.index), n_folds=n_folds)
    # TODO: kan je ook geen confusion matrices genereren?
    tree_confusion_matrices = {}

    for train, test in kf:
        X_train = DataFrame(features_df, index=train)
        X_test = DataFrame(features_df, index=test)
        y_train = DataFrame(labels_df, index=train)
        y_test = DataFrame(labels_df, index=test)

        rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        rf.fit(X_train.values.tolist(), y_train['cat'].tolist())
        predicted_labels = []
        for index, vector in enumerate(X_test.values):
            predicted_labels.append(rf.predict(vector.reshape(1, -1))[0])

        y_test_np = y_test['cat']._values
        predicted_labels_np = np.array(predicted_labels)
        y_test_np = np.array(y_test_np)
        equal_positions = 1.0 * np.sum(predicted_labels_np == y_test_np)
        accuracy = equal_positions / len(predicted_labels)
        print accuracy

# TODO: De run code moet hier weg
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('data/heart.dat', sep=' ')
# df = df.iloc[np.random.permutation(len(df))]
# df = df.reset_index(drop=True)
df.columns = columns

features_column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral',
                         'fasting blood sugar', \
                         'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
                         'number of vessels', 'thal']
# labels_column_names = 'disease'
column_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
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
features_column_names = features_df.columns

evaluate_trees(features_df=features_df, labels_df=labels_df, n_folds=2)
