__author__ = 'manuj'

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


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

#print Xtrain
#print Ytrain

#print Xtest
#print Ytest

XtrainT=[]
XtestT=[]


for array in Xtrain:

    array2=[1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2), array[0]*array[1], abs(array[0]-array[1]), abs(array[0]+array[1])]

    XtrainT.append(array2)

XtrainT= np.array(XtrainT)

for array in Xtest:

    array2=[1, array[0], array[1], math.pow(array[0],2), math.pow(array[1],2), array[0]*array[1], abs(array[0]-array[1]), abs(array[0]+array[1])]

    XtestT.append(array2)

XtestT= np.array(XtestT)


regr = linear_model.LinearRegression()

regr.fit(XtrainT, Ytrain,2)



in_mul = (regr.predict(XtrainT)*Ytrain)
in_sum_error= (in_mul<0).sum()

#print in_mul
#print in_sum_error

out_mul = (regr.predict(XtestT)*Ytest)
out_sum_error= (out_mul<0).sum()


#print out_mul
#print out_sum_error

print "in sample error= " + str(float(in_sum_error)/Ytrain.shape[0])
print "out of sample error= " + str(float(out_sum_error)/Ytest.shape[0])

t1=time.time()-t0

print "run time="+ str(t1)

