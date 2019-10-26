#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import pickle

# read data

X_train = pd.read_csv('data/X_train')
Y_train = pd.read_csv('data/Y_train', header=None)
X_test = pd.read_csv('data/X_test')

print('Y = 0: ', len(Y_train[Y_train[0]==0]) )
print('Y = 1: ', len(Y_train[Y_train[0]==1]) )
print(len(X_train))

# there is a little label imbalance in data
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, Y_train)
# suffling makes performance better and steady
# X_shu, y_shu = shuffle( X_res, y_res )

model = GradientBoostingClassifier()

# scores = cross_val_score(model, X_shu, y_shu, cv=10, scoring='accuracy')
# scores_n = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')

# print('Train with SMOTE and GradientBoostingClassifier' ) 
# print(scores)
# print(scores.mean())
# print('No SMOTE')
# print(scores_n)
# print(scores_n.mean())

print(model)

model.fit(X_train, Y_train)

# save model

filename = 'GBclassfier.pickle'
pickle.dump(model, open(filename, 'wb'))
