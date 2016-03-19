from pandas import DataFrame, read_csv
import pandas as pd
from sklearn import cross_validation

from sklearn.cluster import k_means
from sklearn.cross_validation import KFold
from sklearn.feature_selection import chi2, f_classif

from constructors.treeconstructor import TreeConstructor
import numpy as np
import scipy
from decisiontree import DecisionTree

#ftp://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/13.0/TREE-QUEST.pdf


class QuestConstructor(TreeConstructor):

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

    def __init__(self):
        pass

    def get_name(self):
        return "QUEST"

    def all_feature_vectors_equal(self, training_feature_vectors):
        return len(training_feature_vectors.index) == (training_feature_vectors.duplicated().sum() + 1)


    def pearson_chi_square_test(self, data, feature):
        # First we construct a frequency matrix: F(i, j) = how many samples of class j have feature = i
        unique_values = np.unique(data[feature])
        unique_categories = np.unique(data['cat'])
        frequency_matrix = {}
        expected_matrix = {}

        # For each possible value of feature
        for feature_value in unique_values:
            frequency_row = {}
            # For each possible category
            for category in unique_categories:
                frequency_row[category] = len(data[(data.cat == category) & (data[feature] == feature_value)])
            frequency_matrix[feature_value] = frequency_row

        # Calculate total sum, map(sum, freq_matrix) will return a sum for each list in the matrix
        sum([sum([w for w in v.itervalues()]) for v in frequency_matrix.itervalues()])
        total_sum = sum([sum([w for w in v.values()]) for v in frequency_matrix.values()])

        # Calculated expected frequency for each category and feature value
        for feature_value in unique_values:
            expected_row = {}
            for category in unique_categories:
                expectancy_num = sum(frequency_matrix[feature_value].values())*sum([v[category] for v in frequency_matrix.values()])
                expected_row[category] = float(expectancy_num / total_sum)
            expected_matrix[feature_value] = expected_row

        # Calculate chi_square (using expected_ and frequency_matrix)
        chi_square = 0.0

        for feature_value in unique_values:
            for category in unique_categories:
                if expected_matrix[feature_value][category] != 0:
                    chi_square += (frequency_matrix[feature_value][category] - expected_matrix[feature_value][
                        category]) ** 2 / expected_matrix[feature_value][category]

        # Return the p-value of the chi-squared score
        return scipy.stats.chi2.sf(chi_square, (len(unique_values)-1)*(len(unique_categories)-1))

    def anova_f_test(self, data, feature):
        # Construct frequency matrix and count how many times each class occurs in the data
        unique_values = np.unique(data[feature])
        unique_categories = np.unique(data['cat'])
        frequency_matrix = {}
        occurence_per_category = {}
        number_of_samples = len(data.index)
        sample_mean_per_category = {}

        for category in unique_categories:
            occurence_per_category[category] = len(data[data.cat == category])

        # For each possible value of feature
        for feature_value in unique_values:
            frequency_row = {}
            # For each possible category
            for category in unique_categories:
                frequency_row[category] = len(data[(data.cat == category) & (data[feature] == feature_value)])
            frequency_matrix[feature_value] = frequency_row

        for category in unique_categories:
            sample_mean_per_category[category] = sum([v*frequency_matrix[v][category] for v in frequency_matrix.keys()])/occurence_per_category[category]

        sample_mean = sum(data[feature])/number_of_samples

        f_score_num = 0.0
        for category in unique_categories:
            f_score_num += (occurence_per_category[category] * (sample_mean_per_category[category]-sample_mean)**2)/(len(unique_categories)-1)

        f_score_denum = 0.0
        for value, cat in data[[feature, 'cat']].values:
            f_score_denum += (value-sample_mean_per_category[cat])**2 / (number_of_samples - len(unique_categories))

        f_score = float(f_score_num / f_score_denum)

        return scipy.stats.f.sf(f_score, len(unique_categories)-1, number_of_samples - len(unique_categories))


    def levene_f_test(self, data):
        # For each feature and each class, calculate the mean per class
        feature_columns = data.columns[:-1]
        unique_categories = np.unique(data['cat'])
        mean_per_feature_and_class = {}
        for feature in feature_columns:
            feature_mean_per_class = {}
            for category in unique_categories:
                data_feature_cat = data[(data.cat == category)][feature]
                feature_mean_per_class[category] = float(sum(data_feature_cat)/len(data_feature_cat))
            mean_per_feature_and_class[feature] = feature_mean_per_class

        # Then tranform all the data (sample_point - mean)
        for feature in feature_columns:
            data[feature] = data[[feature, 'cat']].apply((lambda x: abs(x[0] - mean_per_feature_and_class[feature][x[1]])), axis=1)

        return f_classif(data[feature_columns], np.ravel(data['cat']))

    def cross_validation(self, data, k):
        return KFold(len(data.index), n_folds=k, shuffle=True)

    def divide_data(self, data, feature, value):
        """
        Divide the data in two subsets, thanks pandas
        :param data: the dataframe to divide
        :param feature: on which column of the dataframe are we splitting?
        :param value: what threshold do we use to split
        :return: node: initialised decision tree object
        """
        return DecisionTree(left=DecisionTree(data=data[data[feature] <= value]),
                            right=DecisionTree(data=data[data[feature] > value]),
                            label=feature,
                            data=data,
                            value=value)

    def construct_tree(self, training_feature_vectors, labels, default=1, max_nr_nodes=1, discrete_thresh=5, alpha=0.1):
        # First find the best split feature
        feature, type = self.find_split_feature(training_feature_vectors.copy(), labels.copy(),
                                                discrete_thresh=discrete_thresh, alpha=alpha)

        # Can be removed later
        if len(labels) == 0:
            return DecisionTree(label=default, value=None, data=None)

        data = DataFrame(training_feature_vectors.copy())
        data['cat'] = labels

        if feature is None or len(training_feature_vectors.index) <= max_nr_nodes or len(np.unique(data['cat'])) == 1\
                or self.all_feature_vectors_equal(training_feature_vectors):
            # Create leaf most occuring class
            #print(np.bincount(data['cat']))
            label = np.argmax(np.bincount(data['cat'].values.astype(int)))
            return DecisionTree(label=label.astype(str), value=None, data=data)

        split_point = self.find_best_split_point(data.copy(), feature, type)
        split_node = self.divide_data(data, feature, split_point)

        #print(feature, split_point, len(split_node.left.data.index), len(split_node.right.data.index))


        node = DecisionTree(label=split_node.label, value=split_node.value, data=split_node.data)
        node.left = self.construct_tree(split_node.left.data.drop('cat', axis=1),
                                        split_node.left.data[['cat']], default, max_nr_nodes, discrete_thresh, alpha)
        node.right = self.construct_tree(split_node.right.data.drop('cat', axis=1),
                                         split_node.right.data[['cat']], default, max_nr_nodes, discrete_thresh, alpha)

        return node

        # Find the split point

    def find_split_feature(self, training_feature_vectors, labels, discrete_thresh=5, alpha=0.1):
        """
        Construct a tree from a given array of feature vectors
        :param training_feature_vectors: a pandas dataframe containing the features
        :param labels: a pandas dataframe containing the labels in the same order
        :return: decision_tree: a DecisionTree object
        """

        if len(labels) == 0:
            return None, None
        cols = training_feature_vectors.columns

        data = DataFrame(training_feature_vectors.copy())
        data['cat'] = labels

        # Split dataframe in continuous and discrete features:
        continuous_features = []
        discrete_features = []
        for feature in cols.values:
            if len(np.unique(data[feature])) > discrete_thresh:
                continuous_features.append(feature)
            else:
                discrete_features.append(feature)

        """
        chi2_values = []
        for discrete_feature in discrete_features:
            chi2_values.append(self.pearson_chi_square_test(data, discrete_feature))

        anova_f_values = []
        for continuous_feature in continuous_features:
            anova_f_values.append(self.anova_f_test(data, continuous_feature))
        """

        if len(discrete_features) > 0:
            chi2_scores, chi2_values = chi2(training_feature_vectors[discrete_features], np.ravel(labels.values))
        else:
            chi2_values = []
        if len(continuous_features) > 0:
            anova_f_scores, anova_f_values = f_classif(training_feature_vectors[continuous_features], np.ravel(labels.values))
        else:
            anova_f_values = []

        chi2_values = np.where(np.isnan(chi2_values), 1, chi2_values)
        anova_f_values = np.where(np.isnan(anova_f_values), 1, anova_f_values)

        #print discrete_features, chi2_values
        #print continuous_features, anova_f_values

        conc = np.concatenate([chi2_values, anova_f_values])
        conc_features = np.concatenate([discrete_features, continuous_features])
        best_feature_p_value = min(conc)
        best_feature = conc_features[np.argmin(conc)]

        if best_feature_p_value < alpha/len(cols.values):
            if best_feature in continuous_features:
                return best_feature, QuestConstructor.CONTINUOUS
            else:
                return best_feature, QuestConstructor.DISCRETE
        else:
            continuous_features_cat = [item for sublist in [continuous_features, ["cat"]] for item in sublist]
            if len(continuous_features) == 0:
                return None, None
            levene_scores, levene_values = self.levene_f_test(data[continuous_features_cat].copy())
            best_feature_p_value = min(levene_values)
            best_feature = continuous_features[np.argmin(levene_values)]
            if best_feature_p_value < alpha/(len(cols.values)+len(continuous_features)):
                if best_feature in continuous_features:
                    return best_feature, QuestConstructor.CONTINUOUS
                else:
                    return best_feature, QuestConstructor.DISCRETE
            else:
                return None, None

    def find_best_split_point(self, data, feature, type):
        unique_categories = np.unique(data['cat'])
        feature_mean_var_freq_per_class = []
        max_value = []
        min_value = []

        for category in unique_categories:
            data_feature_cat = data[(data.cat == category)][feature]
            feature_mean_var_freq_per_class.append([float(np.mean(data_feature_cat)), float(np.var(data_feature_cat)),
                                                     len(data_feature_cat), category])
            max_value.append(np.max(data_feature_cat))
            min_value.append(np.min(data_feature_cat))

        max_value = np.max(max_value)
        min_value = np.min(min_value)

        # First we transform the discrete variable to a continuous variable and then apply same QDA
        if type == QuestConstructor.DISCRETE:
            data_feature_all_cats = pd.get_dummies(data[feature])
            dummies = data_feature_all_cats.columns
            data_feature_all_cats['cat'] = data['cat']
            mean_freq_per_class_dummies = []
            for category in unique_categories:
                data_feature_cat = data_feature_all_cats[(data.cat == category)]
                mean_freq_per_class_dummies.append([data_feature_cat.as_matrix(columns=dummies).mean(0), len(data_feature_cat.index)])
            overall_mean = data_feature_all_cats.as_matrix(columns=dummies).mean(0)
            #print(mean_freq_per_class_dummies, overall_mean)
            split_point = 0
            # For each class we construct an I x I matrix (with I number of variables), some reshaping required
            B_temp = ([np.dot(np.transpose(np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1))),
                         np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1)))
                  for i in range(len(mean_freq_per_class_dummies))])
            B = B_temp[0]
            for i in range(1, len(B_temp)):
                B = np.add(B, B_temp[i])

            T_temp = [np.dot(np.transpose(np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i,:], overall_mean), (1, -1))),
                             np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i,:], overall_mean), (1, -1)))
                      for i in range(len(data_feature_all_cats.index))]
            T = T_temp[0]
            for i in range(1, len(T_temp)):
                T = np.add(T, T_temp[i])

            # Perform single value decomposition on T: T = Q*D*Q'
            Q, D, Q_t = np.linalg.svd(T)
            # Make sure we don't do sqrt of negative numbers
            D = [0 if i < 0 else 1/i for i in D]
            D_sqrt = np.sqrt(D)
            # Make sure we don't invert zeroes
            D_sqrt_inv = np.diag([0 if i == 0 else 1/i for i in D_sqrt])

            # Get most important eigenvector of using D
            matrix = np.dot(np.dot(np.dot(np.dot(D_sqrt_inv, Q_t), B), Q), D_sqrt_inv)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            largest_eigenvector = eigenvectors[np.argmax(eigenvalues)]

            # We can now transform all discrete attributes to continous ones!
            discrete_values = data[feature].values
            continous_values = []
            discrete_dummies = data_feature_all_cats[dummies].values
            for i in range(len(discrete_values)):
                new_value = np.dot(np.dot(np.dot(np.reshape(largest_eigenvector, (1, -1)), D_sqrt_inv), Q_t), discrete_dummies[i])
                continous_values.append(new_value[0])
            data[feature] = continous_values

        # Now find the best split point
        means = [i[0] for i in feature_mean_var_freq_per_class]
        variances = [i[1] for i in feature_mean_var_freq_per_class]
        frequencies = [i[2] for i in feature_mean_var_freq_per_class]
        if len(unique_categories) != 2:
            # If all class means are equal, pick the one with highest frequency as superclass A, the others as B
            # Then calculate means and variances of the two superclasses
            if len(np.unique(means)) == 1:
                index_a = np.argmax(frequencies)
                mean_a = means[index_a]
                var_a = variances[index_a]
                freq_a = frequencies[index_a]
                sum_freq = sum(frequencies)
                mean_b = sum([(frequencies[i]*means[i])/sum_freq for i in range(len(means)) if i != index_a])
                var_b = sum([(frequencies[i]*variances[i] + frequencies[i]*(means[i] - mean_b))/sum_freq for i in range(len(means)) if i != index_a])
                freq_b = sum([frequencies[i] for i in range(len(means)) if i != index_a])
                #print(mean_a, mean_b, mean_b, variance_b)

            # Else, apply kmeans clustering to divide the classes in 2 superclasses
            else:
                clusters = k_means(np.reshape(means, (-1, 1)), 2,
                                            n_init=1, init=np.asarray([[np.min(means)], [np.max(means)]]))
                mean_a, mean_b = np.ravel(clusters[0])
                labels = clusters[1]
                indices_a = [i for i in range(len(labels)) if labels[i] == 0]
                indices_b = [i for i in range(len(labels)) if labels[i] == 1]
                sum_freq = sum(frequencies)
                var_a = sum([(frequencies[i]*variances[i] + frequencies[i]*(means[i] - mean_a))/sum_freq for i in indices_a])
                var_b = sum([(frequencies[i]*variances[i] + frequencies[i]*(means[i] - mean_b))/sum_freq for i in indices_b])
                freq_a = sum([frequencies[i] for i in indices_a])
                freq_b = sum([frequencies[i] for i in indices_b])

               # print([mean_a, mean_b], [var_a, var_b], [freq_a, freq_b])

        # If there are only two classes, those are the superclasses already
        else:
            mean_a = means[0]
            mean_b = means[1]
            var_a = variances[0]
            var_b = variances[1]
            freq_a = frequencies[0]
            freq_b = frequencies[1]

        split_point = self.calculate_split_point(mean_a, mean_b, var_a, var_b, freq_a, freq_b, max_value, min_value)

        return split_point

    def calculate_split_point(self, mean_a, mean_b, var_a, var_b, freq_a, freq_b, max_val, min_val):
        if np.min([var_a, var_b]) == 0.0:
            if var_a < var_b:
                if mean_a < mean_b:
                    return mean_a*(1+10**(-12))
                else:
                    return mean_a*(1-10**(-12))
            else:
                if mean_b < mean_a:
                    return mean_b*(1+10**(-12))
                else:
                    return mean_b*(1-10**(-12))
        # Else quadratic discriminant analysis (find X such that P(X, A|t) = P(X, B|t))
        else:
            a = var_a - var_b
            b = 2*(mean_a*var_b - mean_b*var_a)
            prob_a = float(float(freq_a) / float(freq_a + freq_b))
            prob_b = 1 - prob_a
            c = (mean_b**2)*var_a - (mean_a**2)*var_b + 2*var_a*var_b*np.log((prob_a * np.sqrt(var_b))/(prob_b * np.sqrt(var_a)))

            disc = b**2-4*a*c
            if disc == 0:
                x1 = (-b+np.sqrt(disc))/(2*a)
                if x1 < min_val or x1 > max_val:
                    return (mean_a + mean_b)/2
                else:
                    return x1
            elif disc > 0:
                x1 = (-b+np.sqrt(disc))/(2*a)
                x2 = (-b-np.sqrt(disc))/(2*a)
                if abs(x1 - mean_a) < abs(x2 - mean_a):
                    if x1 < min_val or x1 > max_val:
                        return (mean_a + mean_b)/2
                    else:
                        return x1
                else:
                    if x2 < min_val or x2 > max_val:
                        return (mean_a + mean_b)/2
                    else:
                        return x2
            else:
                return (mean_a + mean_b)/2

    def calculate_error_rate(self, tree, testing_feature_vectors, labels, significance):
        pass

    def post_prune(self, tree, testing_feature_vectors, labels, significance=0.125):
        pass

"""
# Read csv into pandas frame
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv('../heart.dat', sep=' ')
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
df.columns = columns


# Seperate the dataframe into a class dataframe and feature dataframe
labels_df = DataFrame()
labels_df['cat'] = df['disease']
df = df.drop('disease', axis=1)
feature_vectors_df = df.copy()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature_vectors_df, labels_df,test_size=0.25)


#feature_vectors_df = feature_vectors_df.drop('number of vessels', axis=1)
#feature_vectors_df = feature_vectors_df.drop('thal', axis=1)
tree_constructor = QuestConstructor()
# tree = tree_constructor.construct_tree(feature_vectors_df, labels_df, np.argmax(np.bincount(play)))
# tree.visualise('../tree')

decision_tree = tree_constructor.construct_tree(X_train, y_train, np.argmax(np.bincount(labels_df['cat'])),
                                                discrete_thresh=5, alpha=0.75)
# decision_tree.visualise("../quest")


predicted_labels = decision_tree.evaluate_multiple(X_test)
# for barf in range(len(train_labels_df.index)):
#     own_decision_tree.
decision_tree.plot_confusion_matrix(y_test['cat'], predicted_labels, normalized=True)
decision_tree.populate_samples(X_train, y_train['cat'].tolist())
decision_tree.visualise('./QUEST')
"""
"""
kf = tree_constructor.cross_validation(feature_vectors_df, 2)

i = 0
for train, test in kf:
    train_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=train)
    test_feature_vectors_df = DataFrame(feature_vectors_df.copy(), index=test)
    train_labels_df = DataFrame(labels_df, index=train)
    test_labels_df = DataFrame(labels_df, index=test)
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree' + str(i), with_pruning_ratio=True)
    frame = DataFrame(test_feature_vectors_df.copy())
    frame['cat'] = test_labels_df.copy()
    print(frame)

    tree_constructor.post_prune(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    tree_constructor.set_error_rate(decision_tree, test_feature_vectors_df.copy(), test_labels_df.copy())
    decision_tree.visualise('../tree_pruned' + str(i), with_pruning_ratio=True)
    print(i)
    i += 1
    """

"""
train_feature_vectors_df = DataFrame(feature_vectors_df, index=)

input_vector = DataFrame()
input_vector['outlook'] = np.asarray([1])
input_vector['temp'] = np.asarray([69])
input_vector['humidity'] = np.asarray([97])
input_vector['windy'] = np.asarray([0])
print(tree.evaluate(input_vector))
"""


# TODO: predict probabilities: http://aaaipress.org/Papers/Workshops/2006/WS-06-06/WS06-06-005.pdf
# TODO                         http://cseweb.ucsd.edu/~elkan/calibrated.pdf

# TODO: pruning

# TODO: multivariate splits possible? Split on multiple attributes at once (in C4.5)
