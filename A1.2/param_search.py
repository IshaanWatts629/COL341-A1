import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

args = sys.argv

# Reading training and testing data

train = pd.read_csv(args[1], index_col = 0)    
test = pd.read_csv(args[2], index_col = 0)

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

    return err,W

lr_backtrack = [0.03,0.1,0.3,1,3]
lr_adaptive = [0.1, 0.3, 1, 3, 10, 30]
lr_constant = [0.01, 0.03, 0.1, 0.3]
batch_size = [1,4,16,64,128,256,512]

def check(lr,case):
    grid = np.zeros((len(lr),len(batch_size)))
    best_loss = float('inf')
    best_lr = -1
    best_batch = -1

    for i in range(len(lr)):
        for j in range(len(batch_size)):

            loss = []
            W = np.zeros((X_train.shape[1], 8))
            iterations = 1

            start = time.time()

            while time.time()-start < 600:
                l,W = batch_grad_descent(X_train,Y_train,W,lr[i],iterations,0.5,0.9,case,batch_size[j])
                loss.append(l) 
                iterations += 1

            print(f"Learning Rate --> {lr[i]}, Batch Size --> {batch_size[j]}, Loss --> {loss[-1]}")
            #plt.plot(loss)
            #plt.ylabel("Negative of Log Likelihood")
            #plt.xlabel("Epochs")
            #plt.show()

            grid[i][j] = loss[-1]

            if loss[-1]<best_loss:
                best_loss = loss[-1]
                best_lr = lr[i]
                best_batch = batch_size[j]


    
    print(f"Best Learning Rate --> {best_lr}, Best Batch Size --> {best_batch},Best Loss --> {best_loss}")
    

print('Constant')
check(lr_constant,1)
print('Adaptive')
check(lr_adaptive,2)
print('Backtrack')
check(lr_backtrack,3)