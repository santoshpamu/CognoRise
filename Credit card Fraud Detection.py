#!/usr/bin/env python
# coding: utf-8

# #### 1. Import Required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\home\Downloads\archive (9).zip")


# In[3]:


df.head()


# In[4]:


df.tail()


# #### 2. Basic  Data Anlysing

# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df['Class'].value_counts() # checkng whether the output class is balanced or imbalanced, unfortunately it is imbalanced target variable


# In[9]:


df['Class'].value_counts(normalize=True) #checking class balance but in percentage format


# #### 3. Data Visualization

# In[10]:


sns.countplot(df['Class'])
plt.title('Class Distribution')
plt.show()


# #### 4. Preprocessing the data

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[27]:


df['Time']=scaler.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount']=scaler.fit_transform(df['Time'].values.reshape(-1,1))


# In[29]:


X=df.drop('Class',axis=1)
y=df['Class']
x_train,y_train,x_test,y_yext=train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',x_test.shape)


# ### As the target variable is imbalanced we  need to balance the target variable into same count by using under sampling we can do that

# In[44]:


def under_sampling(df):
    legit=df[df['Class']==0]
    fraud=df[df['Class']==1]
    legit_sample=legit.sample(n=len(fraud))
    new_df=pd.concat([legit_sample,fraud],axis=0)
    print(new_df.head())
    print(new_df['Class'].value_counts())
    X=new_df.drop(columns='Class',axis=1)
    y=new_df['Class']    
    return X,y
    
    
X_sample, y_sample = under_sampling(df)
    


# #### 5. Splitting the data into train and test data

# In[46]:


X_train,X_test,y_train,y_test=train_test_split(X_sample,y_sample,test_size=0.2,random_state=42)


# In[47]:


# Print the shapes of the datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# #### 6.Model Building

# In[50]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[51]:


rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)


# #### 7. Predicting on Unseen data

# In[52]:


y_test_pred=rfc.predict(X_test)


# In[56]:


y_test_pred


# In[57]:


temp_df=pd.DataFrame({'Actual':y_test,'Predicted':y_test_pred})
temp_df.head()


# #### 8.checing accuracy by Evaluation metrics

# In[58]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)


# In[59]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[60]:


y_test_pred=lr.predict(X_test)


# In[61]:


y_test_pred


# In[62]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)


# In[ ]:




