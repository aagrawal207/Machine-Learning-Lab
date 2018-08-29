# Assignment 1

Showcasing Linear Regression, Regularized Linear Regression and Logistic Regression.

### Linear Regression

In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables.

### Regularized Linear Regression

It is similar Linear Regression. It is used to prevent over-fitting on the training data. Here we normalize the weights of the hypothesis function.

### Logistic Regression

Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.

### Dependencies

We used python3 for writing the algorithm and were required these modules:
1. Numpy
1. Pandas
1. Matplotlib
1. Scikit Learn
1. PrettyTable

Running .ipynb notebooks using Jupyter Nootbook is preferred.

### Usage

### Code

The assignment code is divided into 2 files `linear-and-regularized-linear-regression.ipynb` and `logistic-regression.ipynb`. There equivalent python files are also present with the same name and .py extension.

#### How to run?

You can either run the jupyter notebooks by opening the .ipynb files which is preferred. Or you can also run their equivalent python files using python3.

### Results

#### Linear Regression

1. Predicted alcohol content from learning on the 5-fold cross validation can seen in the jupyter notebook.

1. Average residual sum of squares for the 5 folds when cross validation is applied over the 5 folds is `128.714415`.

#### Regularized Linear Regression

1. Residual Error for each fold is

```
+-------------+--------------------+
| fold number | mean squared error |
+-------------+--------------------+
|      1      | 0.7858459796909978 |
|      2      | 0.8709725065053622 |
|      3      | 0.6744291887499511 |
|      4      | 1.0477968331955168 |
|      5      | 0.6765498077097857 |
+-------------+--------------------+
```
.

1. The value of lambda which gives the best result is `0.000100`.

#### Logistic Regression

1. & 3.

```
Fold number 1
+-------+--------------------+--------------------+---------------------+
| Label |     Precision      |       Recall       |        Fscore       |
+-------+--------------------+--------------------+---------------------+
|   3   |        0.0         |        0.0         |         0.0         |
|   4   |        0.0         |        0.0         |         0.0         |
|   5   | 0.5511111111111111 | 0.9051094890510949 |  0.6850828729281768 |
|   6   | 0.5217391304347826 |       0.375        | 0.43636363636363634 |
|   7   |        0.6         |       0.075        | 0.13333333333333333 |
|   8   |        0.0         |        0.0         |         0.0         |
+-------+--------------------+--------------------+---------------------+

Confusion Matrix
[[  0   0   2   0   0   0]
 [  0   0   8   3   0   0]
 [  0   0 124  13   0   0]
 [  0   0  80  48   0   0]
 [  0   0  11  26   3   0]
 [  0   0   0   2   2   0]]
************************************




Fold number 2
+-------+---------------------+--------------------+--------------------+
| Label |      Precision      |       Recall       |       Fscore       |
+-------+---------------------+--------------------+--------------------+
|   3   |         0.0         |        0.0         |        0.0         |
|   4   |         0.0         |        0.0         |        0.0         |
|   5   |  0.5813953488372093 | 0.7352941176470589 | 0.6493506493506493 |
|   6   | 0.46923076923076923 |     0.4765625      | 0.4728682170542636 |
|   7   | 0.42105263157894735 |        0.2         | 0.2711864406779661 |
|   8   |         0.0         |        0.0         |        0.0         |
+-------+---------------------+--------------------+--------------------+

Confusion Matrix
[[  0   0   1   1   0   0]
 [  0   0   9   2   0   0]
 [  0   0 100  33   3   0]
 [  0   0  61  61   6   0]
 [  0   0   1  31   8   0]
 [  0   0   0   2   2   0]]
************************************




Fold number 3
+-------+--------------------+--------------------+--------------------+
| Label |     Precision      |       Recall       |       Fscore       |
+-------+--------------------+--------------------+--------------------+
|   3   |        0.0         |        0.0         |        0.0         |
|   4   |        0.0         |        0.0         |        0.0         |
|   5   | 0.6748466257668712 | 0.8088235294117647 | 0.7357859531772575 |
|   6   | 0.5364238410596026 |     0.6328125      | 0.5806451612903226 |
|   7   | 0.2857142857142857 |        0.05        | 0.0851063829787234 |
|   8   |        0.0         |        0.0         |        0.0         |
+-------+--------------------+--------------------+--------------------+

Confusion Matrix
[[  0   0   2   0   0   0]
 [  0   0   5   6   0   0]
 [  0   0 110  25   1   0]
 [  0   0  44  81   3   0]
 [  0   0   2  36   2   0]
 [  0   0   0   3   1   0]]
************************************




Fold number 4
+-------+--------------------+--------------------+---------------------+
| Label |     Precision      |       Recall       |        Fscore       |
+-------+--------------------+--------------------+---------------------+
|   3   |        0.0         |        0.0         |         0.0         |
|   4   |        0.0         |        0.0         |         0.0         |
|   5   | 0.7241379310344828 | 0.6176470588235294 |  0.6666666666666667 |
|   6   | 0.5263157894736842 | 0.7874015748031497 |  0.6309148264984228 |
|   7   |        0.5         |        0.15        | 0.23076923076923075 |
|   8   |        0.0         |        0.0         |         0.0         |
+-------+--------------------+--------------------+---------------------+

Confusion Matrix
[[  0   0   2   0   0   0]
 [  0   0   6   4   0   0]
 [  0   0  84  52   0   0]
 [  0   0  22 100   5   0]
 [  0   0   2  32   6   0]
 [  0   0   0   2   1   0]]
************************************




Fold number 5
+-------+--------------------+--------------------+--------------------+
| Label |     Precision      |       Recall       |       Fscore       |
+-------+--------------------+--------------------+--------------------+
|   3   |        0.0         |        0.0         |        0.0         |
|   4   |        0.0         |        0.0         |        0.0         |
|   5   |       0.704        | 0.6470588235294118 | 0.6743295019157088 |
|   6   | 0.5208333333333334 | 0.7874015748031497 | 0.6269592476489029 |
|   7   |        0.0         |        0.0         |        0.0         |
|   8   |        0.0         |        0.0         |        0.0         |
+-------+--------------------+--------------------+--------------------+

Confusion Matrix
[[  0   0   2   0   0   0]
 [  0   0   7   3   0   0]
 [  0   0  88  48   0   0]
 [  0   0  27 100   0   0]
 [  0   0   1  38   0   0]
 [  0   0   0   3   0   0]]
```

2.

```
+--------+----------------+----------------+----------------+
| Labels | Avg_Precision  |   Avg_Recall   |   Avg_Fscore   |
+--------+----------------+----------------+----------------+
|   3    |      0.0       |      0.0       |      0.0       |
|   4    |      0.0       |      0.0       |      0.0       |
|   5    | 0.64709820335  | 0.742786603693 | 0.682243128808 |
|   6    | 0.514908572706 | 0.611835629921 | 0.549550217771 |
|   7    | 0.361353383459 |     0.095      | 0.144079077552 |
|   8    |      0.0       |      0.0       |      0.0       |
+--------+----------------+----------------+----------------+
```
