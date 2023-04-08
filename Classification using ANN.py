#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


data = pd.read_csv("Crop_recommendation.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[6]:


X


# In[7]:


y


# In[8]:


X1=data
X1=X1.drop(['label'],axis=1)
X1


# In[9]:


y1=data['label']
# y1
# y1=pd.get_dummies(y1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data['label'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# In[10]:


X1


# In[11]:


y1


# In[12]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y1[:]=le.fit_transform(y1[:])
y1


# In[13]:


from  sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.2,random_state=500)


# In[14]:


type(y_train)


# In[15]:


type(X_train)


# In[16]:


y_train=np.array(y_train)
y_train


# In[17]:


X_train=np.array(X_train)
X_train


# In[18]:


y_train = np.asarray(y_train).astype(np.float32)


# In[19]:


ann=tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=44,activation='relu'))
ann.add(tf.keras.layers.Dense(units=44,activation='relu'))
ann.add(tf.keras.layers.Dense(units=22,activation='softmax'))
ann.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[20]:


ann.fit(X_train,y_train,batch_size=30,epochs=100)


# In[21]:


y_train[0].dtype


# In[22]:


y_train


# In[23]:


y_train = np.asarray(y_train).astype(np.float32)


# In[24]:


X_train


# In[25]:


y_pred=ann.predict(X_test)
y_pred


# In[27]:


from sklearn.preprocessing import StandardScaler as sc

def predict_crop(N,P,K,temp,humidity,ph,rainfall):
    # Scale the values
    sc = sc()
    input_array = np.array([[N,P,K,temp,humidity,ph,rainfall]])
    input_array = sc.fit_transform(input_array)
    # Predict
    prediction = ann.predict(input_array)
    # Return the prediction
    return prediction

