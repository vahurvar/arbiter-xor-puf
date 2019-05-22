#!/usr/bin/env python
# coding: utf-8

# In[77]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import XorPuf as puf

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

import math


# In[78]:


def print_metrics(y_test, y_pred):
    print('Matrix', confusion_matrix(Y_test, y_pred))
    print('Acc', accuracy_score(Y_test, y_pred))
    print('report', classification_report(Y_test, y_pred))


# In[79]:


def train(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = np.round(model.predict(X_test))
    print_metrics(Y_test, pred)


# In[80]:


def calculate_phi(challenges):
        return np.prod([math.pow(-1, c) for c in challenges])
    
def get_XY_phi(data):
    X, Y = [], []
    
    for l in data:
        Y.append(l.pop())
        phi = [calculate_phi(l[i:]) for i in range(len(l))]
        X.append(phi)
    
    return np.asarray(X), np.asarray(Y)


# In[88]:


bits = [5, 8, 10, 12, 15, 18, 20]

for bit in bits:
    print('Bits', bit)
    xor_puf = puf.XorPUF(bit, 5)
    data = xor_puf.calculate_responses()
    print('Data', len(data))
    
    X,Y = get_XY_phi(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(LogisticRegression(), X_train, Y_train, X_test, Y_test)
    print('--------')


# In[82]:


from keras.models import Sequential
from keras.layers import Dense

def get_NN_model(puf_bits):
    model = Sequential()
    model.add(Dense(puf_bits, input_dim=puf_bits, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# In[83]:


epochs = 50
batch_size = 1000


# In[89]:


bits = [5, 8, 10, 12, 15, 18, 20]

for bit in bits:
    print('Bits', bit)
    xor_puf = puf.XorPUF(bit, 5)
    data = xor_puf.calculate_responses()
    X, Y = get_XY_phi(data)
    print('Data', len(X))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = get_NN_model(bit)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle='batch', verbose=0)
    
    print('Loss ', model.evaluate(X_test, Y_test))
    pred = np.round(model.predict(X_test))

    print_metrics(Y_test, pred)
    
    print('--------')


# In[91]:


bits = [32, 64]

for bit in bits:
    print('Bits', bit)
    xor_puf = puf.XorPUF(bit, 5)
    data = xor_puf.calculate_responses_with_random_challenges(10_000)
    X, Y = get_XY_phi(data)
    print('Data', len(X))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = get_NN_model(bit)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle='batch', verbose=0)
    
    print('Loss ', model.evaluate(X_test, Y_test))
    pred = np.round(model.predict(X_test))

    print_metrics(Y_test, pred)
    
    print('--------')


# In[90]:


bits = [32, 64]

for bit in bits:
    print('Bits', bit)
    xor_puf = puf.XorPUF(bit, 5)
    data = xor_puf.calculate_responses_with_random_challenges(500_000)
    X, Y = get_XY_phi(data)
    print('Data', len(X))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = get_NN_model(bit)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle='batch', verbose=0)
    
    print('Loss ', model.evaluate(X_test, Y_test))
    pred = np.round(model.predict(X_test))

    print_metrics(Y_test, pred)
    
    print('--------')


# In[ ]:




