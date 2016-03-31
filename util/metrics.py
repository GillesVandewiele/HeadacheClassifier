import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix'):
    fig = plt.figure()
    cm = np.divide(cm, len(cm))
    cm = np.divide(cm, np.matrix.sum(np.asmatrix(cm))).round(3)

    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(cm, cmap=plt.get_cmap('RdYlGn'))
    for (j, i), label in np.ndenumerate(cm):
        ax.text(i, j, label, ha='center', va='center')
    fig.colorbar(cax)
    plt.show()