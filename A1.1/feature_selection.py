import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures

args = sys.argv

df = pd.read_csv(args[0], index_col=0)

# X,Y training data unprocessed

X = df.drop('Total Costs', axis = 1)
Y = df['Total Costs']

# Normalising X

for col in X.columns:
    mean = np.mean(X[col])
    std = np.std(X[col])
    X[col] = (X[col] - mean)/std

Y_train = Y.to_numpy()

lamda = [0.0001, 0.0003, 0.001, 0.003, 0.01]

def Kfold_score(x,y,lamda,k=10):
    
    score_ = []
    
    m = x.shape[0]
    fold = m//k
    
    for i in range(k):
        x_test = x[i*fold:(i+1)*fold,:]
        x_train = x[:i*fold,:]
        x_train = np.append(x_train, x[(i+1)*fold:,:],0)
        
        y_test = y[i*fold:(i+1)*fold]
        y_train = y[:i*fold]
        y_train = np.append(y_train, y[(i+1)*fold:],0)
        
        lasso = LassoLars(alpha=lamda)
        lasso.fit(x_train,y_train)
        score_.append(lasso.score(x_test,y_test))
        
    return np.mean(score_)	

# Creating polynomial features of degree 2 and choosing best lamda

cols = X.columns

poly = PolynomialFeatures(2, include_bias=False)
X_poly = poly.fit_transform(X_train)
X_new = pd.DataFrame(X_poly, columns = poly.get_feature_names(cols))

cols_new = X_new.columns

X_train_1 = X_new.to_numpy()

score = []
for val in lamda:
    score.append(Kfold_score(X_train_1, Y_train, val))
    
best_lamda = lamda[np.argmax(score)]

#best_lamda = 0.0003

# Feature Selection using Lasso

lasso = LassoLars(best_lamda)
lasso.fit(X_train_1,Y_train)

w = sorted(lasso.coef_, key = lambda x : abs(x), reverse=True)
threshold = abs(w[295])

drop = []

w = lasso.coef_

for i in range(len(w)):
    if abs(w[i])<threshold:
        drop.append(i)

drop_cols = [cols_new[col] for col in drop]

# Final features --> 296

X_final = X_new.drop(drop_cols, axis = 1)
X_final_ = X_final.to_numpy()

Kfold_score(X_final_, Y_train,0.0003)

# Score = 0.6804439503434712


drop = [13, 14, 17, 21, 22, 28, 37, 38, 42, 43, 44, 47, 48, 49, 51, 52, 53, 54, 55, 58, 65, 66, 67, 73, 74, 75, 76, 77, 78, 80, 81, 84, 85, 86, 88, 94, 96, 99, 101, 103, 105, 108, 110, 112, 115, 119, 121, 122, 126, 127, 128, 131, 133, 136, 137, 139, 147, 148, 151, 155, 158, 159, 161, 162, 163, 166, 167, 172, 174, 183, 187, 194, 196, 200, 202, 203, 205, 210, 211, 212, 214, 215, 217, 219, 220, 221, 223, 227, 228, 231, 233, 234, 235, 237, 238, 239, 241, 243, 245, 247, 248, 250, 255, 257, 259, 261, 262, 263, 266, 267, 269, 270, 271, 273, 274, 276, 277, 278, 280, 282, 283, 284, 287, 291, 300, 312, 316, 319, 320, 322, 326, 327, 328, 334, 337, 338, 339, 340, 351, 354, 355, 356, 368, 370, 371, 372, 373, 375, 378, 379, 381, 382, 383, 385, 386, 387, 396, 397, 399, 400, 401, 412, 420, 421, 422, 424, 425, 426, 428, 431, 435, 438, 443, 445, 446, 447, 454, 456, 457, 460, 462, 463, 464, 469, 470, 471, 475, 476, 481, 482, 483, 484, 487, 488, 489, 490, 491, 492, 494]

drop_cols = ['CCS Diagnosis Code',
 'CCS Diagnosis Description',
 'APR DRG Code',
 'APR Severity of Illness Code',
 'APR Severity of Illness Description',
 'Birth Weight',
 'Health Service Area Gender',
 'Health Service Area Race',
 'Health Service Area Patient Disposition',
 'Health Service Area CCS Diagnosis Code',
 'Health Service Area CCS Diagnosis Description',
 'Health Service Area APR DRG Code',
 'Health Service Area APR DRG Description',
 'Health Service Area APR MDC Code',
 'Health Service Area APR Severity of Illness Code',
 'Health Service Area APR Severity of Illness Description',
 'Health Service Area APR Risk of Mortality',
 'Health Service Area APR Medical Surgical Description',
 'Health Service Area Payment Typology 1',
 'Health Service Area Birth Weight',
 'Hospital County Zip Code - 3 digits',
 'Hospital County Gender',
 'Hospital County Race',
 'Hospital County CCS Diagnosis Description',
 'Hospital County CCS Procedure Code',
 'Hospital County CCS Procedure Description',
 'Hospital County APR DRG Code',
 'Hospital County APR DRG Description',
 'Hospital County APR MDC Code',
 'Hospital County APR Severity of Illness Code',
 'Hospital County APR Severity of Illness Description',
 'Hospital County Payment Typology 1',
 'Hospital County Payment Typology 2',
 'Hospital County Payment Typology 3',
 'Hospital County Emergency Department Indicator',
 'Operating Certificate Number Gender',
 'Operating Certificate Number Ethnicity',
 'Operating Certificate Number Patient Disposition',
 'Operating Certificate Number CCS Diagnosis Description',
 'Operating Certificate Number CCS Procedure Description',
 'Operating Certificate Number APR DRG Description',
 'Operating Certificate Number APR Severity of Illness Code',
 'Operating Certificate Number APR Risk of Mortality',
 'Operating Certificate Number Payment Typology 1',
 'Operating Certificate Number Birth Weight',
 'Facility Id Age Group',
 'Facility Id Gender',
 'Facility Id Race',
 'Facility Id Patient Disposition',
 'Facility Id CCS Diagnosis Code',
 'Facility Id CCS Diagnosis Description',
 'Facility Id APR DRG Code',
 'Facility Id APR MDC Code',
 'Facility Id APR Severity of Illness Description',
 'Facility Id APR Risk of Mortality',
 'Facility Id Payment Typology 1',
 'Facility Name Gender',
 'Facility Name Race',
 'Facility Name Type of Admission',
 'Facility Name CCS Procedure Code',
 'Facility Name APR DRG Description',
 'Facility Name APR MDC Code',
 'Facility Name APR Severity of Illness Code',
 'Facility Name APR Severity of Illness Description',
 'Facility Name APR Risk of Mortality',
 'Facility Name Payment Typology 2',
 'Facility Name Payment Typology 3',
 'Age Group Gender',
 'Age Group Ethnicity',
 'Age Group APR DRG Description',
 'Age Group APR Severity of Illness Description',
 'Age Group Emergency Department Indicator',
 'Zip Code - 3 digits Gender',
 'Zip Code - 3 digits Type of Admission',
 'Zip Code - 3 digits CCS Diagnosis Code',
 'Zip Code - 3 digits CCS Diagnosis Description',
 'Zip Code - 3 digits CCS Procedure Description',
 'Zip Code - 3 digits APR Severity of Illness Code',
 'Zip Code - 3 digits APR Severity of Illness Description',
 'Zip Code - 3 digits APR Risk of Mortality',
 'Zip Code - 3 digits Payment Typology 1',
 'Zip Code - 3 digits Payment Typology 2',
 'Zip Code - 3 digits Birth Weight',
 'Gender^2',
 'Gender Race',
 'Gender Ethnicity',
 'Gender Type of Admission',
 'Gender CCS Procedure Code',
 'Gender CCS Procedure Description',
 'Gender APR MDC Code',
 'Gender APR Severity of Illness Code',
 'Gender APR Severity of Illness Description',
 'Gender APR Risk of Mortality',
 'Gender Payment Typology 1',
 'Gender Payment Typology 2',
 'Gender Payment Typology 3',
 'Gender Emergency Department Indicator',
 'Race Ethnicity',
 'Race Type of Admission',
 'Race CCS Diagnosis Code',
 'Race CCS Diagnosis Description',
 'Race CCS Procedure Description',
 'Race APR Severity of Illness Code',
 'Race APR Risk of Mortality',
 'Race Payment Typology 1',
 'Race Payment Typology 3',
 'Race Birth Weight',
 'Race Emergency Department Indicator',
 'Ethnicity Type of Admission',
 'Ethnicity Patient Disposition',
 'Ethnicity CCS Diagnosis Description',
 'Ethnicity CCS Procedure Code',
 'Ethnicity CCS Procedure Description',
 'Ethnicity APR DRG Description',
 'Ethnicity APR MDC Code',
 'Ethnicity APR Severity of Illness Code',
 'Ethnicity APR Severity of Illness Description',
 'Ethnicity APR Risk of Mortality',
 'Ethnicity Payment Typology 1',
 'Ethnicity Payment Typology 3',
 'Ethnicity Birth Weight',
 'Ethnicity Emergency Department Indicator',
 'Length of Stay Patient Disposition',
 'Length of Stay CCS Procedure Description',
 'Length of Stay Payment Typology 1',
 'Type of Admission APR DRG Description',
 'Type of Admission APR Severity of Illness Description',
 'Type of Admission Payment Typology 1',
 'Type of Admission Payment Typology 2',
 'Type of Admission Birth Weight',
 'Patient Disposition CCS Diagnosis Description',
 'Patient Disposition CCS Procedure Code',
 'Patient Disposition CCS Procedure Description',
 'Patient Disposition APR Severity of Illness Description',
 'Patient Disposition Payment Typology 1',
 'Patient Disposition Payment Typology 2',
 'Patient Disposition Payment Typology 3',
 'Patient Disposition Birth Weight',
 'CCS Diagnosis Code APR Severity of Illness Description',
 'CCS Diagnosis Code Payment Typology 1',
 'CCS Diagnosis Code Payment Typology 2',
 'CCS Diagnosis Code Payment Typology 3',
 'CCS Diagnosis Description APR Risk of Mortality',
 'CCS Diagnosis Description Payment Typology 1',
 'CCS Diagnosis Description Payment Typology 2',
 'CCS Diagnosis Description Payment Typology 3',
 'CCS Diagnosis Description Birth Weight',
 'CCS Procedure Code^2',
 'CCS Procedure Code APR DRG Description',
 'CCS Procedure Code APR MDC Code',
 'CCS Procedure Code APR Severity of Illness Code',
 'CCS Procedure Code APR Severity of Illness Description',
 'CCS Procedure Code APR Risk of Mortality',
 'CCS Procedure Code Payment Typology 1',
 'CCS Procedure Code Payment Typology 2',
 'CCS Procedure Code Payment Typology 3',
 'CCS Procedure Description APR Severity of Illness Description',
 'CCS Procedure Description APR Risk of Mortality',
 'CCS Procedure Description Payment Typology 1',
 'CCS Procedure Description Payment Typology 2',
 'CCS Procedure Description Payment Typology 3',
 'APR DRG Code Payment Typology 1',
 'APR DRG Description APR Severity of Illness Code',
 'APR DRG Description APR Severity of Illness Description',
 'APR DRG Description APR Risk of Mortality',
 'APR DRG Description Payment Typology 1',
 'APR DRG Description Payment Typology 2',
 'APR DRG Description Payment Typology 3',
 'APR DRG Description Emergency Department Indicator',
 'APR MDC Code APR Severity of Illness Code',
 'APR MDC Code Payment Typology 1',
 'APR MDC Code Birth Weight',
 'APR MDC Description APR Risk of Mortality',
 'APR MDC Description Payment Typology 1',
 'APR MDC Description Payment Typology 2',
 'APR MDC Description Payment Typology 3',
 'APR Severity of Illness Code Payment Typology 1',
 'APR Severity of Illness Code Payment Typology 3',
 'APR Severity of Illness Code Birth Weight',
 'APR Severity of Illness Description APR Risk of Mortality',
 'APR Severity of Illness Description Payment Typology 1',
 'APR Severity of Illness Description Payment Typology 2',
 'APR Severity of Illness Description Payment Typology 3',
 'APR Risk of Mortality Payment Typology 1',
 'APR Risk of Mortality Payment Typology 2',
 'APR Risk of Mortality Payment Typology 3',
 'APR Medical Surgical Description Payment Typology 1',
 'APR Medical Surgical Description Payment Typology 2',
 'Payment Typology 1 Payment Typology 2',
 'Payment Typology 1 Payment Typology 3',
 'Payment Typology 1 Birth Weight',
 'Payment Typology 1 Emergency Department Indicator',
 'Payment Typology 2 Birth Weight',
 'Payment Typology 2 Emergency Department Indicator',
 'Payment Typology 3^2',
 'Payment Typology 3 Birth Weight',
 'Payment Typology 3 Emergency Department Indicator',
 'Birth Weight^2',
 'Emergency Department Indicator^2']