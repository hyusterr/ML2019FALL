#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import math
import numpy as np
import pandas as pd

t0 = time.time()

dim = 106

# sys.argv

# $1: raw data (train.csv)  
# $2: test data (test.csv)  
# $3: provided train feature (X_train)  
# $4: provided train label (Y_train)
# $5: provided test feature (X_test)     
# $6: ans.csv


# load data and return as np.array

def load_data():
    
    x_train = pd.read_csv( sys.argv[3] )
    x_test = pd.read_csv( sys.argv[5] )

    x_train = x_train.values # copy as np.array
    x_test = x_test.values

    y_train = pd.read_csv( sys.argv[4], header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6) 
    # bound the range of output, there won't be a high 


def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[ 0: x_train.shape[0] ]
    x_test_nor = x_all_nor[ x_train.shape[0]: ]

    return x_train_nor, x_test_nor


def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros( ( dim, ) )
    mu2 = np.zeros( ( dim, ) )
    
    for i in range( x_train.shape[0] ):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros( (dim, dim) )
    sigma2 = np.zeros( (dim, dim) )
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2


def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv( share_sigma )

    w = np.dot( (mu1 - mu2), sigma_inverse)
    b = (-0.5) * np.dot( np.dot( mu1.T, sigma_inverse ), mu1) + (0.5) * np.dot( np.dot( mu2.T, sigma_inverse ), mu2) + np.log( float( N1 ) / N2 )

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

if __name__ == '__main__':
    x_train, y_train, x_test = load_data()
    
    x_train, x_test = normalize(x_train, x_test)
    
    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)

    
    y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
    
    y = np.around(y)
    
    result = (y_train == y)
    
    print('Train accuracy = %f' % (float(result.sum()) / result.shape[0]))


pred_test = predict(x_test, mu1, mu2, shared_sigma, N1, N2)   

lst = [ str(i + 1) + ',' + str( 1 ) + '\n' if pred_test[i] > 0.5 else str(i + 1) + ',' + str( 0 ) + '\n' for i in range( len( pred_test ) ) ]


# print(lst)

with open(sys.argv[6], 'w') as f:
    f.write('id,label\n')
    f.writelines(lst)
    f.close()

print( 'Training time  = ', time.time() - t0 )
