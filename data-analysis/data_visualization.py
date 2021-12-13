import pandas
import sklearn

import data_setup

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def box_and_whisker(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()

def histograms(dataset):
    dataset.hist()
    pyplot.show()

def scatter_plots(dataset):
    scatter_matrix(dataset)
    pyplot.show()

if __name__=='__main__':
    dataset = data_setup.load_data()
    box_and_whisker(dataset)
    histograms(dataset)
    scatter_plots(dataset)
