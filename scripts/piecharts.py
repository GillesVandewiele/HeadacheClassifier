from sklearn import datasets

import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv, DataFrame
import os
import numpy as np

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



columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join(os.path.join('..', 'data'), 'heart.dat'), sep=' ')
df.columns=columns
labels_df = DataFrame()
labels_df['cat'] = df['disease'].copy()

heart_distribution = {}
for value in labels_df['cat'].values:
    if value not in heart_distribution:
        heart_distribution[value] = 1
    else:
        heart_distribution[value] += 1

total_sum = np.sum(heart_distribution.values())
for value in heart_distribution:
    heart_distribution[value] = float(heart_distribution[value]) / float(total_sum)

plot_pie_chart(heart_distribution.values(), heart_distribution.keys(),
               'Distribution of the classes in the heart disease dataset')

columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = read_csv(os.path.join('../data', 'car.data'), sep=',')
df.columns=columns
labels_df = DataFrame()
labels_df['cat'] = df['class'].copy()
car_distribution = {}
for value in labels_df['cat'].values:
    if value not in car_distribution:
        car_distribution[value] = 1
    else:
        car_distribution[value] += 1

total_sum = np.sum(car_distribution.values())
for value in car_distribution:
    car_distribution[value] = float(car_distribution[value]) / float(total_sum)

plot_pie_chart(car_distribution.values(), car_distribution.keys(),
               'Distribution of the classes in the car dataset')

iris = datasets.load_iris()
df = DataFrame(iris.data)
df.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
df["Name"] = np.add(iris.target, 1)



flower_distribution = {}
for value in df["Name"].values:
    if value not in flower_distribution:
        flower_distribution[value] = 1
    else:
        flower_distribution[value] += 1

total_sum = np.sum(flower_distribution.values())
for value in flower_distribution:
    flower_distribution[value] = float(flower_distribution[value]) / float(total_sum)

plot_pie_chart(flower_distribution.values(), flower_distribution.keys(),
               'Distribution of the classes in the iris dataset')


columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

mapping_parents = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
mapping_has_nurs = {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4}
mapping_form = {'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3}
mapping_housing = {'convenient': 0, 'less_conv': 1, 'critical': 2}
mapping_finance = {'convenient': 0, 'inconv': 1}
mapping_social = {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2}
mapping_health = {'recommended': 0, 'priority': 1, 'not_recom': 2}
mapping_class = {'not_recom': 1, 'recommend': 0, 'very_recom': 2, 'priority': 3, 'spec_prior': 4}

df = read_csv(os.path.join('../data', 'nursery.data'), sep=',')
df = df.dropna()
df.columns=columns

df['parents'] = df['parents'].map(mapping_parents)
df['has_nurs'] = df['has_nurs'].map(mapping_has_nurs)
df['form'] = df['form'].map(mapping_form)
df['children'] = df['children'].map(lambda x: 4 if x == 'more' else int(x))
df['housing'] = df['housing'].map(mapping_housing)
df['finance'] = df['finance'].map(mapping_finance)
df['social'] = df['social'].map(mapping_social)
df['health'] = df['health'].map(mapping_health)
df['class'] = df['class'].map(mapping_class)

# df = df[df['class'] != 0]
df = df.reset_index(drop=True)


print len(df)

nurse_distribution = {}
for value in df["class"].values:
    if value not in nurse_distribution:
        nurse_distribution[value] = 1
    else:
        nurse_distribution[value] += 1

total_sum = np.sum(nurse_distribution.values())
for value in nurse_distribution:
    nurse_distribution[value] = float(nurse_distribution[value]) / float(total_sum)

plot_pie_chart(nurse_distribution.values(), nurse_distribution.keys(),
               'Distribution of the classes in the nursery dataset')


columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
           'feature9', 'class' ]
df = read_csv(os.path.join('../data', 'shuttle.tst'), sep=' ')
df.columns=columns
# df = df[df['class'] < 6]
df = df.reset_index(drop=True)
print np.bincount(df['class'])
feature_mins = {}
feature_maxs = {}
feature_column_names = list(set(df.columns) - set(['class']))

for feature in feature_column_names:
    if np.min(df[feature]) < 0:
        df[feature] += np.min(df[feature]) * (-1)
        feature_mins[feature] = 0
    else:
        feature_mins[feature] = np.min(df[feature])

    feature_maxs[feature] = np.max(df[feature])

df=df.reset_index(drop=True)

print len(df)

shuttle_distribution = {}
for value in df["class"].values:
    if value not in shuttle_distribution:
        shuttle_distribution[value] = 1
    else:
        shuttle_distribution[value] += 1

total_sum = np.sum(shuttle_distribution.values())
for value in shuttle_distribution:
    shuttle_distribution[value] = float(shuttle_distribution[value]) / float(total_sum)

print shuttle_distribution

dict = {}
for i in range(1,6):
    dict[i] = shuttle_distribution[i]

dict[6] = shuttle_distribution[6] + shuttle_distribution[7]
print dict

__keys = [1,2,4,3,5,6]
_values  = []
for key in __keys:
    _values.append(dict[key])

print _values
__keys[-1] = "6 & 7"
plot_pie_chart(_values, __keys,
               'Distribution of the classes in the shuttle dataset')

