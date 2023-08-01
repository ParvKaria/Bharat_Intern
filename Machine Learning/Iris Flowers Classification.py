#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


#load the data

df = pd.read_csv('iris.csv')
df.head()


# In[16]:


#some basic statistical analysis about the data

df.describe()


# In[17]:


#basic info of data

df.info()


# In[18]:


#sample of each class

df['Species'].value_counts()


# In[19]:


#check for null values

df.isnull().sum()


# In[23]:


df['SepalLengthCm'].hist()


# In[24]:


df['SepalWidthCm'].hist()


# In[25]:


df['PetalLengthCm'].hist()


# In[26]:


df['PetalWidthCm'].hist()


# In[30]:


#visualize the whole data

sns.pairplot(df, hue='Species')


# In[36]:


# Drop the species column 
df_numeric = df.drop(['Species'], axis=1)

# Calculate correlation on the numeric columns
corr = df_numeric.corr()
corr


# In[38]:


fig,ax=plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot=True, ax=ax, cmap='coolwarm')


# In[39]:


#lable encoding

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[40]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[62]:


# Separate features and target
data = df.values
X = data[:,0:4] 
Y = data[:,4]


# In[76]:


#split the data to train and test data
#train 80
#test 20

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[77]:


# Encode labels 
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species']) 

# Now split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[78]:


y_train = y_train.astype(int) 
y_test = y_test.astype(int)


# In[79]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[82]:


#print performance matrix

print("Accuracy : ", model.score(X_test,y_test))


# In[83]:


# Add regularization to logistic regression
from sklearn.linear_model import LogisticRegressionCV

logreg = LogisticRegressionCV(Cs=10, cv=5, penalty='l2')
logreg.fit(X_train, y_train)

print("Logistic Regression CV Accuracy:", logreg.score(X_test, y_test))

# Use LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train) 

print("LDA Accuracy:", lda.score(X_test, y_test))

# Use k-fold cross validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(logreg, X, y, cv=5)
print("Cross Validation Accuracy:", np.mean(cv_scores))

# Confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", conf_mat)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




