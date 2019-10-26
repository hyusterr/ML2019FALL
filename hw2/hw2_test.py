#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

X_test = pd.read_csv(sys.argv[5]) 

filename = 'GBclassfier.pickle'
model = pickle.load(open(filename, 'rb'))

Y_test = model.predict(X_test)

lst = [str(i + 1) + ',' + str(Y_test[i]) + '\n' for i in range(len(Y_test))]

with open(sys.argv[6], 'w') as f:
    f.write('id,label\n')
    f.writelines(lst)
    f.close()
