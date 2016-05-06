import random
from pandas import DataFrame
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class DataCollector(object):

    def __init__(self):
        pass

    @staticmethod
    def load_data_from_db(host, port, dbname):
        client = MongoClient(host, port)
        db = client[dbname]
        return db

    @staticmethod
    def calculate_age(born):
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    @staticmethod
    def plot_pie_chart(values, labels, title):

        # The slices will be ordered and plotted counter-clockwise.
        my_norm = matplotlib.colors.Normalize(0, 1) # maps your data to the range [0, 1]
        my_cmap = matplotlib.cm.get_cmap('coolwarm')
        print my_norm(values)
        fig = plt.figure()
        fig.suptitle(title, fontsize=25)

        matplotlib.rcParams['font.size'] = 18
        plt.pie(values, labels=labels, colors=my_cmap(my_norm(values)),
                autopct='%1.1f%%')
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0]*1.25, Size[1]*1.75, forward=True)
        plt.show()


database = DataCollector.load_data_from_db('localhost', 9000, 'CHRONIC')

# Get all collections from the database
patients = database['patient']
drugs = database['drug']
headaches = database['headache']
medicines = database['medicine']
symptoms = database['symptom']
triggers = database['trigger']

####################################################
#       Read the patient data into dataframe       #
####################################################
patient_column_names = ['id', 'age', 'sex', 'relation', 'employment', 'diagnosis']
patients_list = []
for patient in patients.find({}):
    patients_list.append([patient['patientID'], patient['birthDate'], patient['isMale'], patient['relation'],
                          patient['isEmployed'], patient['diagnosis']])

####################################################
#   Map strings to integers, fill missing values   #
####################################################
patient_df = DataFrame(patients_list, columns=patient_column_names)
patient_df['age'] = [DataCollector.calculate_age(datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")) if x != "null"
                     else np.NaN for x in patient_df['age']]
patient_df.age.replace(np.NaN, patient_df["age"].mean(), inplace=True)
patient_df['age'] = patient_df['age'].astype(int)

relation_mapping = {"VRIJGEZEL": 0, "IN RELATIE": 1,"GETROUWD": 2}
patient_df['relation'] = patient_df['relation'].map(relation_mapping)
patient_df['relation'] = patient_df.relation.apply(lambda x: x if not pd.isnull(x) else 0)
patient_df['sex'] = patient_df['sex'].map(lambda x: 1 if x else 0)
patient_df['employment'] = patient_df['employment'].map(lambda x: 1 if x else 0)
patient_df['diagnosis'] = patient_df.diagnosis.apply(lambda x: random.randint(0, 4))  # TODO: this ain't how it should be
print patient_df
print patient_df.describe()

###################################################
#           Plot some demographic plots           #
###################################################


# def get_distribution(values):
#     distribution = {}
#     for value in values:
#         if value not in distribution:
#             distribution[value] = 1
#         else:
#             distribution[value] += 1
#
#     total_sum = np.sum(distribution.values())
#     for value in distribution:
#         distribution[value] = float(distribution[value]) / float(total_sum)
#
#     return distribution
#
# categorical_columns = ['sex', 'relation', 'employment', 'diagnosis']
# for column in categorical_columns:
#     col_distribution = get_distribution(patient_df[column].values)
#     DataCollector.plot_pie_chart(col_distribution.values(), col_distribution.keys(),
#                                  'Distribution of the ' + column + ' in the headache dataset')
#
# n, bins, patches = plt.hist(patient_df['age'], 5, normed=0, facecolor='blue', alpha=0.75)
# plt.xlabel('Age')
# plt.ylabel('Relative amount')
# plt.title('Distribution of the age in the headache dataset')
# plt.show()

####################################################
#      Read the headache data into dataframe       #
#   We process the strings already in datatypes    #
####################################################
headache_locations = ["frontal_right", "frontal_mid", "frontal_left", "parietal_right", "parietal_mid",
                      "parietal_left", "temporal_right", "temporal_left", "occipital_right", "occipital_mid",
                      "occipital_left", "cervical_right", "cervical_mid", "cervical_left", "orbital_right",
                      "orbital_left", "mandibular_left", "mandibular_right", "maxillar_right", "maxillar_left"]
headache_column_names = ['id', 'intensities', 'end', 'patientID', 'symptomIDs', 'triggerIDs', 'locations']
headaches_list = []
for i in range(len(patient_df)):
    for headache in headaches.find({"patientID": patient_df.iloc[i, :]['id']}):
        row = [headache['headacheID']]
        intensity_dict = {}
        for intensity in headache['intensityValues']:
            intensity_dict[datetime.strptime(intensity['key'], "%Y-%m-%dT%H:%M:%S.%fZ")] = int(intensity['value'])
        row.append(intensity_dict)
        # Missing value for end: add 2 hours
        row.extend([datetime.strptime(headache['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
                    if headache['end'] != "null" else sorted(list(intensity_dict.keys()))[0] + timedelta(hours=2),
                    headache['patientID'], headache['symptomIDs'], headache['triggerIDs']])
        location_dict = {}
        for location in headache['locations']:
            location_dict[location['key']] = location['value']
        row.append(location_dict)
        headaches_list.append(row)

headache_df = DataFrame(headaches_list, columns=headache_column_names)
print headache_df
print headache_df.describe()

####################################################
#    Now that we have all required information,    #
#       we can make a features dataframe           #
####################################################

data_list = []

for i in range(len(patient_df)):
    vector = []
    # All patient demographic attributes are features
    vector.extend(patient_df.iloc[i, :].values)

    # Count number of headaches for a patient
    vector.append(len(headache_df[headache_df.patientID == patient_df.iloc[i,:]['id']]))
    filtered_df = headache_df[headache_df.patientID == patient_df.iloc[i,:]['id']]
    intensity_values = []
    durations = []
    location_freq_dict = {}
    for location in headache_locations:
        location_freq_dict[location]=0
    for _headache in range(len(filtered_df)):
        headache = filtered_df.iloc[_headache, :]
        intensity_values.extend(headache['intensities'].values())
        duration = (headache['end'] - sorted(list(headache['intensities'].keys()))[0]).total_seconds()
        if duration < 0:
            duration = 7200
            #TODO: interpolate  sorted(list(headache['intensities'].items()))
        durations.append(duration)
        for location in headache['locations'].items():
            location_freq_dict[location[0]] += location[1]

    # Intensity value mean and max
    vector.append(np.mean(intensity_values))
    vector.append(np.max(intensity_values) if len(intensity_values) else np.NaN)

    # Duration mean, max and min
    vector.append(np.mean(durations))
    vector.append(np.max(durations) if len(durations) else np.NaN)
    vector.append(np.min(durations) if len(durations) else np.NaN)

    # Relative frequency of all intensity values (0, 1, .., 10)
    vector.extend(np.histogram(intensity_values, bins=range(12), normed=True)[0])

    # Relative frequency for each location
    total_sum = sum(location_freq_dict.values())
    for location in headache_locations:
        if total_sum > 0:
            vector.append(location_freq_dict[location]/total_sum)
        else:
            vector.append(0)
    data_list.append(vector)

intensity_names = []
for i in range(11):
    intensity_names.append("intensity_"+str(i))
columns = ["id", "age", "sex", "relation", "employment", "diagnosis", "headacheCount", "meanIntensity", "maxIntensity",
           "meanDuration", "maxDuration", "minDuration"]
columns.extend(intensity_names)
columns.extend(headache_locations)

data_df = DataFrame(data_list, columns=columns)

print data_df
features_df = data_df.copy()
features_df = features_df.dropna()
features_df = features_df.drop('diagnosis', axis=1)
pca = PCA(n_components=2)
pca.fit(features_df)

transformed_features = []
for i in range(len(features_df)):
    feature_vector = features_df.iloc[i, :]
    transformed_feature = [feature_vector['id']]
    transformed_feature.extend(*pca.transform(feature_vector))
    transformed_features.append(transformed_feature)

print transformed_features
transformed_features_df = DataFrame(transformed_features, columns=["id", "pca_1", "pca_2"])
print transformed_features_df

fig, ax = plt.subplots()

ax.scatter(transformed_features_df['pca_1'], transformed_features_df['pca_2'])

for i in range(len(transformed_features_df)):
    feature_vector = transformed_features_df.iloc[i, :]
    ax.annotate(feature_vector['id'], (feature_vector['pca_1'], feature_vector['pca_2']))

plt.show()

db = DBSCAN(eps=15000, min_samples=1).fit(transformed_features_df[['pca_1', 'pca_2']])

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# data_df = data_df.dropna()
#
# clusters = fclusterdata(data_df[["meanIntensity", "meanDuration"]], 0.1, criterion="distance")
# print clusters