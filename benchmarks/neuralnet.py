import matplotlib.pyplot as plt
import os
from pandas import read_csv, DataFrame

import lasagne
import numpy as np
import sys
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, PrintLayerInfo, BatchIterator, TrainSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix


def build_nn(nr_features):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            # ('dropout1', layers.dropout),
            # ('hidden2', layers.DenseLayer),
            # ('dropout2', layers.dropout),
            # ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # # layer parameters
        # l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
        #                              input_var=input_var)

        # input_shape=(None, 1, 28, 28),
        input_shape=(None,nr_features),
        hidden_num_units=80,
        # hidden2_num_units=500,
        # hidden3_num_units=200,
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        # batch_iterator_train=BatchIterator(batch_size=16),
        # objective_loss_function=lasagne.objectives.squared_error,
        output_num_units=2,
        regression=False,
        # train_split=TrainSplit(eval_size=0.1),
        # dropout1_p=0.8,
        # dropout2_p=0.5,
        # optimization method:
        # hidden_non_linearity = lasagne.nonlinearities.softmax,
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=8000,
        verbose=0,
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
        print y_train
        print "Train feature vectors shape: " + X_train.shape.__str__()
        print "Train labels shape:" + len(y_train).__str__()

        print "X_train as array shape: " + str(X_train.shape)
        print "y_train as array shape: " + str(np.reshape(np.asarray(y_train), (-1, 1)).shape)

        model.fit(X_train, np.reshape(np.asarray(y_train), (-1, 1)).ravel())

        preds = model.predict(X_test)
        c = []
        [c.append(preds[i]) if preds[i] == y_test[i] else None for i in range(min(len(y_test), len(preds)))]
        # checks = len([i for i, j in zip(preds, np.reshape(np.asarray(y_train), (-1, 1))) if i == j])

        model = None
        del model
        # cm = confusion_matrix(y_test, preds)
        # plt.matshow(cm)
        # plt.title('Confusion matrix')
        # plt.colorbar()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()

        print preds.tolist()
        print "number of ones: " + str(sum(preds))
        print y_test
        print c
        print ((len(c) * 1.0) / (len(y_test) * 1.0))

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
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
features_df = (features_df-features_df.mean())/(features_df.max()-features_df.min())
features_column_names = features_df.columns

local_test(feature_vectors_df=features_df, labels_df=labels_df, k=10)
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# model = build_nn(nr_features=X_train.shape[1])
# model.initialize()
# layer_info = PrintLayerInfo()
# layer_info(model)
#
# # Fit our model
#
# y_train = np.reshape(np.asarray(y_train, dtype='int32'), (-1, 1)).ravel()
# print y_train
# print "Train feature vectors shape: " + X_train.shape.__str__()
# print "Train labels shape:" + len(y_train).__str__()
#
# print "X_train as array shape: " + str(X_train.shape)
# print "y_train as array shape: " + str(np.reshape(np.asarray(y_train), (-1, 1)).shape)
#
# model.fit(X_train, np.reshape(np.asarray(y_train), (-1, 1)).ravel())
#
# preds = model.predict(X_test)
# c = []
# [c.append(preds[i]) if preds[i] == y_train[i] else None for i in range(min(len(y_train), len(preds)))]
# # checks = len([i for i, j in zip(preds, np.reshape(np.asarray(y_train), (-1, 1))) if i == j])
# print preds
# print ((len(c) * 1.0) / (len(y_test) * 1.0))
