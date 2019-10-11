#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import math
import numpy as np
import pandas as pd
import re
import pickle

def readdata(data):

    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values
    data = np.delete(data, [0,1], 1)
    
    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)
    
    return data


def extract(data):
    N = data.shape[0] // 18
    # print(N)
    temp = data[:18, :]
    
    # Shape 會變成 (x, 18) x = 取多少hours
    for i in range(1, N):
        temp = np.hstack((temp, data[i*18: i*18+18, :]))
    return temp


def valid(x, y):
    if y <= 2 or y > 100:
        return False
    for i in range(9):
        if x[9,i] <= 2 or x[9,i] > 100:
            return False
    return True


def parse2train(data):
    x = []
    y = []
    

    total_length = data.shape[1] - 9
    for i in range(total_length):
        x_tmp = data[:,i:i+9]
        y_tmp = data[9,i+9]
        if valid(x_tmp, y_tmp):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    
    x = np.array(x)
    y = np.array(y)
    
    return x,y


pattern = re.compile('[0-9\.]')
def parser(string):
#     print(string)
    if string == 'NR':
        return float(0)
    elif type(string) == str:
#         print(string)
        s = ""
        for i in string:
            if re.match(pattern, i):
                s += i
        if s == "":
            return float('nan')
        else:
            return float(s)
    else:
        return string

test = pd.read_csv( sys.argv[1] )

Test = []
idlist = []
for i, group in test.groupby('id'):
    x = []
    idlist.append(group.iloc[0,0])
    for idx, row in group.iterrows():
        x += list(row.drop(['id', '測項']).map(parser))
    Test.append(x)

with open('model.pickle', 'rb') as file:
    w, bias =pickle.load(file)


ans = []
for i in np.array(pd.DataFrame(Test).fillna(0)):
    ans.append((np.dot(w.reshape(162,), i) + bias))

pd.concat((pd.DataFrame(idlist), pd.DataFrame(ans)), axis=1).to_csv(sys.argv[2], index=False, header=['id', 'value'])

