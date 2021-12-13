# IRIS DATASET

This project uses the well-known Fisher iris classification dataset (found [here](https://archive.ics.uci.edu/ml/datasets/Iris) and [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv)).
The dataset was used to train a machine learning model to determine the class of irises (Iris Setosa, Versicolour, or Virginica) 
based on 4 attributes: sepal length, sepal width, petal length, and petal width; all measured in cm.

As noted in the UCI archive, one of the classes can be distinguished linearly from the other two, but the other two cannot be
linearly distinguished from one another. 

The particular dataset has 150 entries, 50 of each class. As provided, the data has errors in the petal width of entry 35, and
has errors in the sepal width and petal length of entry 38 (noted in the UCI archive link provided).

## Table of Contents

- [Configuration](#configuration)
- [Data Overview](#data-overview)

## Configuration
This project was done entirely in python, using the following versions and libraries:
```
Python: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
scipy: 1.7.1
numpy: 1.20.3
matplotlib: 3.4.3
pandas: 1.3.4
sklearn: 0.24.2
```

## Data Overview
### Basic Information
Before getting into the project, I took a look at the data overall, since it wouldn't be very easy to analyze it or have
any ideas of what to do with it without having a general idea of how it looks. The following is the output from my 
`data-setup.py` script. 

We can see below that the shape is 150 x 5, as we might expect, since we were promised 150 entries in the data set, and
each one should have the 4 attributes as well as the 1 class that the iris actually belongs to. We can further see this
layout in the sample 20 lines of the data frame that are printed out. Finally, we can see at the very bottom that each
class has 50 entries in the dataset, as promised.

```
Shape: (150, 5)

Sample Data:
    sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa


class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
```

### Visualization
Of course, it helps to visualize the data as well, so my `data_visualization.py` creates
box-and-whisker plots, histograms, and a scatter plot matrix to help understand the data
more visually.

box plots of individual attributes, to see general layout and outliers:

<img src="https://github.com/stephenKaliman/iris-dataset/blob/main/figures/box-and-whisker.png">

histograms of individual attributes, to see shape and more detailed distribution:

<img src="https://github.com/stephenKaliman/iris-dataset/blob/main/figures/histogram.png">

scatter plots of attributes (in pairs) to see how they correlate to one another. This may help to give an idea of what we might be able to do with the data:

<img src="https://github.com/stephenKaliman/iris-dataset/blob/main/figures/scatter.png">

## Modeling
Now, we have to move away from the data science/statistics and towards the machine learning.
In order to actually accomplish anything to that effect, we have to pick an algorithm/model to use.
The pool I chose from was a set of 6 common algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA),
K-Nearest Neighbors (KNN), Classiciation and Regression Trees (CART), Gaussian Naive Bayes (GNB), and Support Vector Machines (SVM).
### Testing
I used stratified k-fold cross validation to test each of these models on the data set.
K-fold cross validation involves splitting the test data into k parts, training on k-1
of these, and testing on the remaining part. This is repeated k times, with each part being
the test part for one of the times. I used a k-value of 10, since this number (apparently) tends to give
consistently reliable results for validation in predicting how the model will hold up in general, based on the given data.

I also did the k-fold validation multiple times with multiple splits in the data to get the following results:
```
LR: 0.949167 (0.058998)
LDA: 0.976528 (0.041023)
KNN: 0.955417 (0.054779)
CART: 0.947986 (0.060929)
NB: 0.950069 (0.060486)
SVM: 0.981597 (0.037150)
```
The label for each of these is the model (see above). The first number is the average performance (average # correct out of 12 across multiple k-fold cross validations) and the second is the standard deviation. As we can see, the model that seems to have performed  best on this data is support vector machines, with an average accuraccy of 98% and a standard deviation of about 3.7%. The standard deviation isn't great, but it's the lowest we have, and the average is the highest we have. So only using this small data set and this naive set of algorithms and fairly basic testing schema, this seems like the best we can go, so for the time being, we'll go on ahead with it.


