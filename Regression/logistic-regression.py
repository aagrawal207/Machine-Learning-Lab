
# coding: utf-8

# In[34]:


import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from prettytable import PrettyTable


# In[2]:


df = pd.read_csv('winequality-red.csv', delimiter=';')


# In[4]:


output = df['quality']


# In[5]:


inp = df.drop('quality', axis=1)


# In[6]:


X = inp.values
y = output.values


# In[69]:


def my_custom_score(estimator, X, y):
    global avg_numpy, fold_num
    table = PrettyTable(['Class', 'Precision', 'Recall', 'Fscore'])
    y_pred = estimator.predict(X)
    scores = precision_recall_fscore_support(
        y, y_pred, average=None, warn_for=())

    avg_numpy = np.add(scores, avg_numpy)

    precision_per_class = scores[0].tolist()
    recall_per_class = scores[1].tolist()
    fscore_per_class = scores[2].tolist()

    print("Fold number {0}".format(fold_num+1))
    fold_num += 1
    for label in range(6):
        table.add_row([label+3, precision_per_class[label],
                       recall_per_class[label], fscore_per_class[label]])
    print(table)
    print('\nConfusion Matrix\n{0}\n************************************\n'.format(
        confusion_matrix(y, y_pred, labels=[3, 4, 5, 6, 7, 8])))
    print("\n\n")
    return 1


# In[70]:


model = LogisticRegression()
avg_numpy = np.zeros((4, len(set(y.tolist()))))
fold_num = 0
scores = cross_val_score(model, X, y, scoring=my_custom_score, cv=5)
avg_numpy = np.divide(avg_numpy, 5)


# In[65]:


table = PrettyTable(['Class', 'Avg_Precision', 'Avg_Recall', 'Avg_Fscore'])


# In[66]:


for label in range(6):
    table.add_row([label + 3, avg_numpy[0][label],
                   avg_numpy[1][label], avg_numpy[2][label]])
print(table)
