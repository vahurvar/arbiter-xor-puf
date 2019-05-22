#!/usr/bin/env python
# coding: utf-8

# In[9]:


from PUF import PUF
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import math


# In[10]:


def train(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_pred = np.heaviside(model.predict(X_test), 0)        
    print(accuracy_score(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))


# In[11]:


def get_XY(data):
    X, Y = [], []
    
    for d in data:
        Y.append(d.pop())
        X.append(d)
    
    return np.asarray(X), np.asarray(Y)


# In[13]:


bits = [5, 8, 10, 12, 15, 20]

for bit in bits:
    print('Bits', bit)
    data = PUF(bit).calculate_responses()
    print('Data', len(data))
    
    X,Y = get_XY(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(LogisticRegression(), X_train, Y_train, X_test, Y_test)
    print('--------')


# In[14]:


bits = [32, 64, 128]
data = [10_000, 10_000, 100_000]

for bit, d in zip(bits,data):
    print('Bits', bit)
    data = PUF(bit).calculate_responses_with_random_challenges(d)
    print('Data', len(data))
    
    X,Y = get_XY(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(LogisticRegression(), X_train, Y_train, X_test, Y_test)
    print('--------')


# In[6]:


def calculate_phi(challenges):
        return np.prod([math.pow(-1, c) for c in challenges])
    
def get_XY_phi(data):
    X, Y = [], []
    
    for d in data:
        Y.append(d.pop())
        phi = [calculate_phi(d[i:]) for i in range(len(d))]
        X.append(phi)
    
    return np.asarray(X), np.asarray(Y)


# In[15]:


bits = [5, 8, 10, 12, 15, 20]

for bit in bits:
    print('Bits', bit)
    data = PUF(bit).calculate_responses()
    print('Data', len(data))
    
    X,Y = get_XY_phi(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(LogisticRegression(), X_train, Y_train, X_test, Y_test)
    print('--------')


# In[18]:


bits = [32, 64, 128]
data = [10_000, 10_000, 100_000]

for bit,d  in zip(bits, data):
    print('Bits', bit)
    data = PUF(bit).calculate_responses_with_random_challenges(d)
    print('Data', len(data))
    
    X,Y = get_XY_phi(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(LogisticRegression(), X_train, Y_train, X_test, Y_test)
    print('--------')


# In[ ]:




