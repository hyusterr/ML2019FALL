#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import numpy as np
import pandas as pd


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
    
    # initialize parameters
    
    b = 0.0 
    w = np.zeros( x_train.shape[1] )
    lr = 0.05
    epoch = 1000
    
    # apply adagrad

    b_lr = 0
    w_lr = np.ones( x_train.shape[1] )
    
    for e in range( epoch ):
        z = np.dot( x_train, w ) + b # vector
        pred = sigmoid( z )
        loss = y_train - pred

        # calculate gradient

        b_grad = -1 * np.sum( loss )
        w_grad = -1 * np.dot( loss, x_train )

        b_lr += b_grad ** 2
        w_lr += w_grad ** 2


        b = b - lr / np.sqrt(b_lr) * b_grad
        w = w - lr / np.sqrt(w_lr) * w_grad

        if (e + 1) % 50 == 0 or e == 0:

            # loss is cross-entropy
            loss = -1 * np.mean( y_train * np.log( pred + 1e-100 ) + ( 1 - y_train ) * np.log( 1 - pred + 1e-100 ))
            print('epoch: {}\tloss: {}'.format( e + 1, loss ) )
    
    return w, b


t0 = time.time()

if __name__ == '__main__':
    x_train, y_train, x_test = load_data()
    
    x_train, x_test = normalize(x_train, x_test)
    
    w, b = train(x_train, y_train)
    
    #predict x_test    

print('Consume time is', time.time() - t0, 'sec.')


z_test = np.dot(x_test, w) + b
pred_test = sigmoid(z_test)

lst = [ str(i + 1) + ',' + str( 1 ) + '\n' if pred_test[i] > 0.5 else str(i + 1) + ',' + str( 0 ) + '\n' for i in range( len( pred_test ) ) ]

with open(sys.argv[6], 'w') as f:
    f.write( 'id,label\n' )
    f.writelines( lst )
    f.close()
