__author__ = 'manuj'

import time
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import cross_validation

t0=time.time()



def load_data(file_name):

    file = open(file_name)

    in_data_array=[]

    for line in file.readlines():
        new_array=line.split()
        in_data_array.append(new_array)


    in_data_array=np.array(in_data_array)

    in_data_array=in_data_array[:,np.newaxis]

    X1= map(float,in_data_array[:,0,0])
    X2= map(float,in_data_array[:,0,1])

    X=np.c_[X1,X2]
    X=np.array(X)

    Y= map(float,in_data_array[:,0,2])
    Y=np.array(Y)

    return(X,Y)


Xtrain,Ytrain= load_data("datain")

Xtest,Ytest= load_data("dataout")



def trans3(array):

    return ([1, array[0], array[1], math.pow(array[0],2)])

def trans4(array):

    return ([1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2)])


def trans5(array):

    return ([1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2), array[0]*array[1]])


def trans6(array):

    return ([1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2), array[0]*array[1], abs(array[0]-array[1])])

def trans7(array):

    return ([1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2), array[0]*array[1], abs(array[0]-array[1]), abs(array[0]+array[1])])

Xtrain3=[trans3(array) for array in Xtrain]
Xtrain4=[trans4(array) for array in Xtrain]
Xtrain5=[trans5(array) for array in Xtrain]
Xtrain6=[trans6(array) for array in Xtrain]
Xtrain7=[trans7(array) for array in Xtrain]

Xtrain3=np.array((Xtrain3))
Xtrain4=np.array((Xtrain4))
Xtrain5=np.array((Xtrain5))
Xtrain6=np.array((Xtrain6))
Xtrain7=np.array((Xtrain7))


Xtest3=[trans3(array) for array in Xtest]
Xtest4=[trans4(array) for array in Xtest]
Xtest5=[trans5(array) for array in Xtest]
Xtest6=[trans6(array) for array in Xtest]
Xtest7=[trans7(array) for array in Xtest]

Xtest3=np.array((Xtest3))
Xtest4=np.array((Xtest4))
Xtest5=np.array((Xtest5))
Xtest6=np.array((Xtest6))
Xtest7=np.array((Xtest7))



n_samples=Xtrain3.shape[0]

cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size=0.2851, random_state=0)

clf3=Ridge(alpha=0,fit_intercept=False)
clf4=Ridge(alpha=0,fit_intercept=False)
clf5=Ridge(alpha=0,fit_intercept=False)
clf6=Ridge(alpha=0,fit_intercept=False)
clf7=Ridge(alpha=0,fit_intercept=False)

scores3 = cross_validation.cross_val_score(clf3, Xtrain3, Ytrain, cv=cv)
scores4 = cross_validation.cross_val_score(clf3, Xtrain4, Ytrain, cv=cv)
scores5 = cross_validation.cross_val_score(clf3, Xtrain5, Ytrain, cv=cv)
scores6 = cross_validation.cross_val_score(clf3, Xtrain6, Ytrain, cv=cv)
scores7 = cross_validation.cross_val_score(clf3, Xtrain7, Ytrain, cv=cv)

print ("\n Q1-----")

print scores3.mean()
print scores4.mean()
print scores5.mean()
print scores6.mean()
print scores7.mean()

print ("\n Q2-----")


clf3.fit(Xtrain3,Ytrain)
clf4.fit(Xtrain4,Ytrain)
clf5.fit(Xtrain5,Ytrain)
clf6.fit(Xtrain6,Ytrain)
clf7.fit(Xtrain7,Ytrain)

print( float ( (np.array(clf3.predict(Xtest3)*Ytest) < 0).sum() ) )/Ytest.shape[0]
print( float ( (np.array(clf4.predict(Xtest4)*Ytest) < 0).sum() ) )/Ytest.shape[0]
print( float ( (np.array(clf5.predict(Xtest5)*Ytest) < 0).sum() ) )/Ytest.shape[0]
print( float ( (np.array(clf6.predict(Xtest6)*Ytest) < 0).sum() ) )/Ytest.shape[0]
print( float ( (np.array(clf7.predict(Xtest7)*Ytest) < 0).sum() ) )/Ytest.shape[0]





#print in_mul
#print in_sum_error

#out_mul = (reg.predict(XtestT)*Ytest)
#ut_sum_error= (out_mul<0).sum()


#print out_mul
#print out_sum_error

#print "in sample error= " + str(float(in_sum_error)/Ytrain.shape[0])
#print "out of sample error= " + str(float(out_sum_error)/Ytest.shape[0])



t1=time.time()-t0

print "run time="+ str(t1)


