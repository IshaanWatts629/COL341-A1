import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

args = sys.argv

def Kfold_loss(x,y,lamda,k=10):
    
    loss = []
    
    m = x.shape[0]
    fold = m//k
    
    for i in range(k):
        x_test = x[i*fold:(i+1)*fold,:]
        x_train = x[:i*fold,:]
        x_train = np.append(x_train, x[(i+1)*fold:,:],0)
        
        y_test = y[i*fold:(i+1)*fold]
        y_train = y[:i*fold]
        y_train = np.append(y_train, y[(i+1)*fold:],0)
        
        x_T = np.transpose(x_train)
        W = np.dot(np.dot(np.linalg.inv(np.dot(x_T,x_train) + lamda*np.eye(x_train.shape[1])),x_T),y_train)
        
        loss.append(1 - (np.linalg.norm(y_test-np.dot(x_test,W))/np.linalg.norm(y_test)))
        
    return np.mean(loss)


if args[1] == 'a':
    df_train = pd.read_csv(args[2], index_col=0)
    df_train.insert(0,'bias',1)

    data_train = df_train.to_numpy()

    X_train = data_train[:,:-1]
    Y_train = data_train[:,-1]

    X_T = np.transpose(X_train)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X_train)),X_T),Y_train)

    df_test = pd.read_csv(args[3], index_col=0)
    df_test.insert(0,'bias',1)

    X_test = df_test.to_numpy()
    Y_test = np.dot(X_test,W)

    np.savetxt(args[4], Y_test, delimiter = '\n')
    np.savetxt(args[5], W, delimiter = '\n')

if args[1] == 'b':
    df_train = pd.read_csv(args[2], index_col=0)
    df_train.insert(0,'bias',1)

    data_train = df_train.to_numpy()

    X_train = data_train[:,:-1]
    Y_train = data_train[:,-1]
    
    with open(args[4],'r') as file:
        lamda = [float(i) for i in file.read().split(',')]
    
    loss = []
    
    for val in lamda:
        loss.append(Kfold_loss(X_train,Y_train,val))
    
    best_lamda = lamda[np.argmax(loss)]
    
    X_T = np.transpose(X_train)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X_train) + best_lamda*np.eye(X_train.shape[1])),X_T),Y_train)

    df_test = pd.read_csv(args[3], index_col=0)
    df_test.insert(0,'bias',1)

    X_test = df_test.to_numpy()
    Y_test = np.dot(X_test,W)

    np.savetxt(args[5], Y_test, delimiter = '\n')
    np.savetxt(args[6], W, delimiter = '\n')
        
    with open(args[7],'w') as file:
        file.write(str(best_lamda))

def process(X):
    for col in X.columns:
        mean = np.mean(X[col])
        std = np.std(X[col])
        X[col] = (X[col] - mean)/std
        
    cols = X.columns

    X_train = X.to_numpy()
    
    poly = PolynomialFeatures(2, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_new = pd.DataFrame(X_poly, columns = poly.get_feature_names(cols))

    cols_new = X_new.columns
    drop = [13, 14, 17, 21, 22, 28, 37, 38, 42, 43, 44, 47, 48, 49, 51, 52, 53, 54, 55, 58, 65, 66, 67, 73, 74, 75, 76, 77, 78, 80, 81, 84, 85, 86, 88, 94, 96, 99, 101, 103, 105, 108, 110, 112, 115, 119, 121, 122, 126, 127, 128, 131, 133, 136, 137, 139, 147, 148, 151, 155, 158, 159, 161, 162, 163, 166, 167, 172, 174, 183, 187, 194, 196, 200, 202, 203, 205, 210, 211, 212, 214, 215, 217, 219, 220, 221, 223, 227, 228, 231, 233, 234, 235, 237, 238, 239, 241, 243, 245, 247, 248, 250, 255, 257, 259, 261, 262, 263, 266, 267, 269, 270, 271, 273, 274, 276, 277, 278, 280, 282, 283, 284, 287, 291, 300, 312, 316, 319, 320, 322, 326, 327, 328, 334, 337, 338, 339, 340, 351, 354, 355, 356, 368, 370, 371, 372, 373, 375, 378, 379, 381, 382, 383, 385, 386, 387, 396, 397, 399, 400, 401, 412, 420, 421, 422, 424, 425, 426, 428, 431, 435, 438, 443, 445, 446, 447, 454, 456, 457, 460, 462, 463, 464, 469, 470, 471, 475, 476, 481, 482, 483, 484, 487, 488, 489, 490, 491, 492, 494]
    drop_cols = [cols_new[col] for col in drop]

    X_final = X_new.drop(drop_cols, axis = 1)
    return X_final.to_numpy()


if args[1] == 'c':
    df_train = pd.read_csv(args[2], index_col=0)

    X = df_train.drop('Total Costs', axis = 1)
    Y = df_train['Total Costs']

    Y_train = Y.to_numpy()
    X_train = process(X)

    X_T = np.transpose(X_train)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X_train)),X_T),Y_train)

    df_test = pd.read_csv(args[3], index_col=0)

    X_test = process(df_test)
    Y_test = np.dot(X_test,W)

    np.savetxt(args[4], Y_test, delimiter = '\n')


        



