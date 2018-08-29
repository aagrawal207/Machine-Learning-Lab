
# coding: utf-8

# In[39]:


import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from matplotlib import pyplot as plt
from prettytable import PrettyTable


# In[2]:


df = pd.read_csv('./Regression/winequality-red.csv', delimiter=';')


# In[3]:


# In[4]:


output = df['alcohol']


# In[5]:


inp = df.drop('alcohol', axis=1)


# In[6]:


X = inp.values  # converting dataframe to numpy array
y = output.values


# In[17]:


def residual_sum_of_squares(estimator, X, y):
    print(
        "Predicted values:\n{0}\n*****************************************\n".format(estimator.predict(X)))
    ans = sum((estimator.predict([X[i]]) - y[i])**2 for i in range(len(y)))
    return ans


# In[18]:


model = LinearRegression()
scores = cross_val_score(model, X, y, scoring=residual_sum_of_squares, cv=5)


# In[19]:


# All scorer objects follow the convention that higher return values are better than lower return values.
# Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, are available as neg_mean_squared_error which return the negated value of the metric.
# print(scores)
print("Average RSS: %f" % (sum(scores)/len(scores)))


# # Question 2

# In[30]:


inp = []
out = []
for alpha in np.arange(0, 1, 0.0001):
    model = Ridge(alpha=alpha)
    inp.append(alpha)
    out.append(np.average(cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=5))*-1)


# In[31]:


# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[32]:


plt.plot(inp, out)
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.title("Graph Error vs Lambda")
plt.show()


# In[33]:


temp = dict(zip(inp, out))


# In[34]:


temp = sorted(temp.items(), key=lambda x: x[1])
print("Value of lambda with least error: %f\nError value: %f" %
      (temp[0][0], temp[0][1]))


# ## Since the error is min for lambda = 0.001, we use this to report the residual error

# In[46]:


model = Ridge(alpha=temp[0][1])
best_scores = cross_val_score(
    model, X, y, scoring='neg_mean_squared_error', cv=5)


# In[47]:


table = PrettyTable(['fold number', 'mean squared error'])


# In[48]:


for i, val in enumerate(best_scores):
    table.add_row([i+1, val*-1])
print(table)
