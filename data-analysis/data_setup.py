import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Load libraries
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

def load_data():
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    # Fix known errors
    dataset.loc[34,'petal-width'] = 0.2
    dataset.loc[37,'sepal-width'] = 3.6
    dataset.loc[37,'petal-length'] = 1.4

def process_data():
    # shape
    print('Shape: '+str(dataset.shape)+"\n")


    # head -- peek at the data
    print("Sample Data:")
    print(dataset.head(20))
    print("\n")


    # class distribution
    print(dataset.groupby('class').size())

if __name__ == '__main__':
    load_data()
    process_data()
