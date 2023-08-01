#!/usr/bin/env python
# coding: utf-8

# In[18]:


# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[19]:


mnist = fetch_openml('mnist_784')


# In[20]:


x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
plt.show()


# In[22]:


x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36005]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
plt.show()


# In[38]:


x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36000]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
plt.show()


# In[32]:


# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)


# In[37]:


# Split into train and test
x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# Shuffle training data
# Shuffle rows 
shuffle_index = np.random.permutation(len(x_train))
x_train = x_train.iloc[shuffle_index]
y_train = y_train.iloc[shuffle_index]

# Create binary labels
y_train = (y_train == '2') 
y_test = (y_test == '2')

# Train model
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Evaluate accuracy 
print('Accuracy :',clf.score(x_test[:100], y_test[:100]))


# In[36]:


# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print('Mean :',a.mean())


# In[ ]:




