import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from pandas import read_csv, DataFrame

from sklearn.cross_validation import KFold
from nolearn.lasagne import NeuralNet, PrintLayerInfo


def build_nn(nr_features):
        net1 = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('dropout1', layers.DropoutLayer),
                ('hidden', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
            # layer parameters
            input_shape=(None, nr_features),
            hidden_num_units=8,
            hidden2_num_units=4,
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=2,

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.03,
            update_momentum=0.90,

            max_epochs=200,
            verbose=1,
        )
        return net1

def local_test(feature_vectors_df, labels_df, k=2):
        kf = KFold(len(feature_vectors_df.index), n_folds=k, shuffle=True, random_state=1337)
        # kf = KFold(500, n_folds=k, shuffle=True, random_state=1337)
        train_errors = []
        test_errors = []

        for train, test in kf:
            # Divide the train_images in a training and validation set (using KFold)
            X_train = DataFrame(feature_vectors_df, index=train)
            X_test = DataFrame(feature_vectors_df, index=test)
            y_train = DataFrame(labels_df, index=train)
            y_test = DataFrame(labels_df, index=test)

            # Logistic Regression for feature selection, higher C = more features will be deleted

            # Feature selection/reduction


            model = build_nn(nr_features=X_train.shape[1])
            new_feature_vectors = np.asarray(X_train.values)
            train_set_results = np.asarray(y_train.values[:].astype('int32'))
            model.initialize()
            layer_info = PrintLayerInfo()
            layer_info(model)

            # Fit our model
            model.__setattr__('allow_input_downcast', True)
            model.fit(new_feature_vectors, train_set_results.ravel())
            model.predict()

columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('heart.dat', sep=' ')
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

local_test(feature_vectors_df=features_df, labels_df=labels_df, k=2)