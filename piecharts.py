from sklearn import datasets

import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv, DataFrame
import os
import numpy as np

def plot_pie_chart(values, labels, title):

    # The slices will be ordered and plotted counter-clockwise.
    my_norm = matplotlib.colors.Normalize(0, 1) # maps your data to the range [0, 1]
    my_cmap = matplotlib.cm.get_cmap('jet')
    print my_norm(values)
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)

    plt.pie(values, labels=labels, colors=my_cmap(my_norm(values)),
            autopct='%1.1f%%')
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show()



columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
           'resting electrocardio', 'max heartrate', 'exercise induced angina', 'oldpeak', 'slope peak', \
           'number of vessels', 'thal', 'disease']
df = read_csv(os.path.join('data', 'heart.dat'), sep=' ')
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
df = read_csv(os.path.join('data', 'car.data'), sep=',')
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

