#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('winequality.csv')
print(df.head())


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[6]:


df.isnull().sum()


# In[7]:


for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()


# In[8]:


df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[39]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[12]:


non_numeric_cols = df.select_dtypes(exclude=['number'])
print(non_numeric_cols)


# In[13]:


non_numeric_cols = df.select_dtypes(exclude=['number'])
df = df.drop(non_numeric_cols, axis=1) 

plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[14]:


df = df.drop('total sulfur dioxide', axis=1)


# In[15]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# In[16]:


df.replace({'white': 1, 'red': 0}, inplace=True)


# In[17]:


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape


# In[18]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[24]:


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()


# In[38]:


from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(ytest, y_pred)

# Plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn

plt.figure()
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


# In[36]:


print(metrics.classification_report(ytest,models[1].predict(xtest)))

