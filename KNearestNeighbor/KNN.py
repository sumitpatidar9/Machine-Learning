#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    distance = np.sqrt( np.sum ( (x1 - x2)**2 ) )
    return distance


class KNN:
    def __init__(self, k = 3):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
            
        
    def predict(self,X):
        predictions = [ self._predict(x) for x in X ]
        return predictions
        
        
    def _predict(self, x):
        #compute the distance
        distances = [ euclidean_distance(x, X_train) for X_train in self.X_train ]
            
            
        #get the colosest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [ self.y_train[i] for i in k_indices ]
            
            
        #majority values
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


# In[5]:


get_ipython().system('jupyter nbconvert --to script KNN.ipynb')


# In[ ]:




