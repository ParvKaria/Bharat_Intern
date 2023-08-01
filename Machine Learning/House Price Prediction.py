#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


HouseDF = pd.read_csv('housing.csv')
HouseDF.head() 


# In[3]:


HouseDF.info()


# In[4]:


HouseDF.describe()


# In[5]:


HouseDF.columns


# In[6]:


sns.pairplot(HouseDF)


# In[7]:


sns.distplot(HouseDF['Price'])


# In[17]:


HouseDF['Price'] = pd.to_numeric(HouseDF['Price'], errors='ignore')


# In[18]:


HouseDF = HouseDF.fillna(0)


# In[19]:


print(HouseDF.applymap(np.isreal).all())


# In[20]:


HouseDF = HouseDF.select_dtypes(include=[np.number])


# In[21]:


corr = HouseDF.corr()
sns.heatmap(corr, annot=True)


# In[22]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 


# In[24]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[25]:


print(lm.intercept_)


# In[29]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


# In[30]:


predictions = lm.predict(X_test)  


# In[31]:


plt.scatter(y_test,predictions)


# In[36]:


sns.distplot((y_test-predictions),bins=50); 


# In[34]:


from sklearn import metrics

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predictions)) 
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions)) 
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




