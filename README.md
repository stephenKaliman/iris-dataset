# IRIS DATASET

This project uses the well-known Fisher iris classification dataset (found [here](https://archive.ics.uci.edu/ml/datasets/Iris) and [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv)).
The dataset was used to train a machine learning model to determine the class of irises (Iris Setosa, Versicolour, or Virginica) 
based on 4 attributes: sepal length, sepal width, petal length, and petal width; all measured in cm.

I used a k-fold cross validation method to choose the SVM model for this particular project. I trained it on 80% of the given data, and tested it on the other 20%, where it performed with 97% accuracy!

As noted in the UCI archive, one of the classes can be distinguished linearly from the other two, but the other two cannot be
linearly distinguished from one another. 

The particular dataset has 150 entries, 50 of each class. As provided, the data has errors in the petal width of entry 35, and
has errors in the sepal width and petal length of entry 38 (noted in the UCI archive link provided).

## Table of Contents

- [Configuration](#configuration)
- [Data Overview](#data-overview)
- [Modeling](#modeling)

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
LR: 0.950278 (0.057774)
LDA: 0.976944 (0.040408)
KNN: 0.958472 (0.051930)
CART: 0.949722 (0.063336)
NB: 0.950000 (0.060381)
SVM: 0.981319 (0.037477)
```
And boxplots to give a more detailed overview of the performance of each model:
<img src = "https://github.com/stephenKaliman/iris-dataset/blob/main/figures/comparison.png">

Note that results vary trial-to-trial due to the use of random divisions to create the k-fold analysis, but these results are fairly representative of multiple trials.

The label for each of these is the model (see above). The first number is the average performance (average # correct out of 12 across multiple k-fold cross validations) and the second is the standard deviation. As we can see, the model that seems to have performed  best on this data is support vector machines, with an average accuraccy of 98% and a standard deviation of about 3.7%. The standard deviation isn't great, but it's the lowest we have, and the average is the highest we have. So only using this small data set and this naive set of algorithms and fairly basic testing schema, this seems like the best we can go, so for the time being, we'll go on ahead with it.

#### Support Vector Machines
Since we are using SVM, we might as well know the basic idea of what it is. In general, it's a popular classification method used in machine learning. Think of any instance of the problem we are trying to solve in terms of the attributes we are using to classify it, in the following way:

We can imagine each instance of the problem as a point plotted in some n-dimensional space, where n is the number of attributes, and the coordinates of the point are the values of the attribute. Then, we can "color" each instance in the space based on its true classification. 

Now, with this in mind, if we want to classify these points, we might want to divide the space up so that different classifications fall in different divisions, so that when we get new data or attempt to classify new data, we can see pretty quickly and easily what classification it most likely will fall into, simply based on how we have divided up the plane. 

SVM does this, and in particular, it picks divisions to maximize the distance from the nearest point in the training set. In doing so, we create a fitting division of the data into separate classifications, and also leave the biggest possible margin of error for any data we might have to classify later, to maximize the probability that we classify it correctly.

As you can see in the images below (credits to [Rohith Gandhi](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)), any of the green lines on the left properly divide the data into categories, but the one on the right gives the largest margin of error for any instances we might encounter when actually using the data later on.

<img src="https://github.com/stephenKaliman/iris-dataset/blob/main/figures/0_9jEWNXTAao7phK-5.png"><img src="https://github.com/stephenKaliman/iris-dataset/blob/main/figures/0_0o8xIA4k3gXUDCFU.png">

Now, of course, leaving the biggest possible margin of error might not always work-- for example, if one of the classes is very well-represented in the data set, then cases we encounter later on will probably not be outliers relative to our data set, while other classes might cross over the "maximum-margin-of-error" division and be incorrectly classified by an SVM model. However, it appears that SVM works quite well for our particular case.

### Predicting
Finally, we have to use our model to make predictions. We used the SVM algorithm, which we decided on in the previous step. We split the data into training and validation parts (80% for training, 20% for validation) since it is already quite small and we don't have extra data readily available. 
#### Results
```
0.9666666666666667
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30
```
So we get an overall accuracy of about 97%, which is pretty good!

## Improvements
Of course, as this stands, it is a finished product in a sense, but it's not completely perfect. I certainly have some reservations about it as it stands, and some plans to improve into the future. This section (and the whole readme and project, but especially this section) is constantly being updated as I work on the project, fix things, and think of new things to improve.

The data set is quite small, but I'm sure I can find more data elsewhere to expand it and have a better separation of training and validation sets, and have each set be larger for more thorough-ness. And besides improving the bare rigor of the project, it might certainly make this a bit more interesting for me (and maybe for you, too) overall if the training and validation sets are completely separate- i.e., they come from different sources, potentially causing more discrepancies in the data to make the whole thing feel a little less contrived and more real, giving me more interesting difficulties to work around.

I noticed that when I changed the seed for the division into training and validation sets, the results from the model testing changed drastically. I plan to use this as an opportunity to explore using the other algorithms, and possibly add more algorithms to the set from which I am testing. But on top of that I feel like the true solution to the problem I see here is to look into better model/algorithm selection techniques.

## References
Inspired by, and loosely followed Jason Brownlee's [introductory example](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)

Some extra information on [k-fold cross verification](https://machinelearningmastery.com/k-fold-cross-validation/)

A basic overview of [SVM](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

And, of course, good old [Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)

