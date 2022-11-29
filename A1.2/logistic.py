import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import SelectKBest,chi2
import time

args = sys.argv

start = time.time()
begin = time.time()

# Reading training and testing data

train = pd.read_csv(args[2], index_col = 0)    
test = pd.read_csv(args[3], index_col = 0)


# Selecting cols for X_train and Y_train 

Y_train = train['Length of Stay']
Y_train = pd.get_dummies(Y_train)
Y_train = np.array(Y_train)  # (m,8)

train = train.drop(columns = ['Length of Stay'])  #(m,25)

# One hot encoding all cols except Total Costs

data = pd.concat([train, test], ignore_index = True)

cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
X_train = data[:train.shape[0], :]
X_test = data[train.shape[0]:, :]

# Adding bias to test and train data

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]



def softmax(A):
    A = A - np.max(A, axis = 1, keepdims=True)
    Z = np.exp(A)
    denom = np.sum(Z, axis = 1).reshape(Z.shape[0],1)
    return Z/denom

def hypothesis(X,W):
    A = np.dot(X,W)
    return softmax(A)

def error(X,Y,W):
    
    m = X.shape[0]

    Y_ = hypothesis(X, W)
    err = np.sum(Y*np.log(Y_))

    return -err/m

def get_grads(X,Y,W):

    m = X.shape[0]
    Y_ = hypothesis(X,W)
    grad_w = np.dot(X.T, Y-Y_)
    return grad_w/m

def grad_descent(X,Y,W,lr,t,alpha,beta,case):
    err = error(X,Y,W)
    grad = get_grads(X,Y,W)

    if case == 2:
        lr = lr/np.sqrt(t)
    elif case == 3:
        while err-error(X, Y, W + lr*grad) < alpha*lr*np.square(np.linalg.norm(grad)):
            lr = lr*beta

    W = W + lr*grad

    return W

def batch_grad_descent(X,Y,W,lr,t,alpha,beta,case,batch_size):

    err = error(X,Y,W)
    m = X.shape[0]

    grad = get_grads(X,Y,W)

    if case == 2:
        lr = lr/np.sqrt(t)

    elif case == 3:
        while err-error(X, Y, W + lr*grad) < alpha*lr*np.square(np.linalg.norm(grad)):
            lr = lr*beta

    for i in range(0,m,batch_size):
        
        if i+batch_size>m:
            break

        X_ = X[i:i+batch_size, :]
        Y_ = Y[i:i+batch_size, :]

        grad_ = get_grads(X_,Y_,W)

        W = W + lr*grad_

    return W

def predict(X,W):

    Y_pred = hypothesis(X,W)
    return np.argmax(Y_pred, axis = 1) + 1

if args[1] == 'a':

    with open(args[4], 'r') as file:
        data = np.loadtxt(file, dtype = 'str')
        case = int(data[0])

        alpha = 0
        beta = 0

        if case == 1 or case == 2:
            lr = float(data[1])

        if case == 3:
            params = [float(i) for i in data[1].split(',')]
            lr = params[0]
            alpha = params[1]
            beta = params[2]

        max_iter = int(data[2])

    W = np.zeros((X_train.shape[1], 8))

    for i in range(max_iter):
        W = grad_descent(X_train,Y_train,W,lr,i+1,alpha,beta,case)

    Y_pred = predict(X_test, W)
    np.savetxt(args[5], Y_pred, delimiter='\n')
    np.savetxt(args[6], W, delimiter='\n')

if args[1] == 'b':

    with open(args[4], 'r') as file:
        data = np.loadtxt(file, dtype = 'str')
        case = int(data[0])

        alpha = 0
        beta = 0

        if case == 1 or case == 2:
            lr = float(data[1])

        if case == 3:
            params = [float(i) for i in data[1].split(',')]
            lr = params[0]
            alpha = params[1]
            beta = params[2]

        max_iter = int(data[2])
        batch_size = int(data[3])


    W = np.zeros((X_train.shape[1], 8))

    for i in range(max_iter):
        W = batch_grad_descent(X_train,Y_train,W,lr,i+1,alpha,beta,case,batch_size)

        if time.time()-start > 17:
            break

    Y_pred = predict(X_test, W)
    np.savetxt(args[5], Y_pred, delimiter='\n')
    np.savetxt(args[6], W, delimiter='\n')

if args[1] == 'c':

    W = np.zeros((X_train.shape[1], 8))

    iterations = 1

    while time.time()-start<550:
        W = batch_grad_descent(X_train,Y_train,W,1,iterations,0.5,0.9,2,64)
        Y_pred = predict(X_test, W)

        iterations += 1
        
        if time.time()-begin > 30:
            np.savetxt(args[4], Y_pred, delimiter='\n')
            np.savetxt(args[5], W, delimiter='\n')
            begin = time.time()

    np.savetxt(args[4], Y_pred, delimiter='\n')
    np.savetxt(args[5], W, delimiter='\n')

if args[1] == 'd':
    feature_select = SelectKBest(chi2, k = 500)
    X_train = feature_select.fit_transform(X_train, Y_train)
    X_test = feature_select.transform(X_test)

    W = np.zeros((X_train.shape[1], 8))

    iterations = 1

    while time.time()-start<800:
        W = batch_grad_descent(X_train,Y_train,W,1,iterations,0.5,0.9,2,64)
        Y_pred = predict(X_test, W)

        iterations += 1

        if time.time()-begin > 30:
            begin = time.time()
            np.savetxt(args[4], Y_pred, delimiter='\n')
            np.savetxt(args[5], W, delimiter='\n')
            
    np.savetxt(args[4], Y_pred, delimiter='\n')
    np.savetxt(args[5], W, delimiter='\n')