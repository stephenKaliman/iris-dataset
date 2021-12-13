import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

import data_setup

from numpy import random
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

def model_testing(dataset):
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4] 
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        cv_results = []
        for i in range(120):
            kfold = StratifiedKFold(n_splits=10, random_state=random.randint(0,2**31), shuffle=True)
            cv_results.append(cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy'))
            results.append(cv_results)
        cv_results = numpy.asarray(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

if __name__ == '__main__':
    dataset = data_setup.load_data()
    model_testing(dataset)
