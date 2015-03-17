__author__ = 'manuj'
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets

t1=time.time()

def load_data(file_name):

    file = open(file_name)

    in_data_array=[]

    for line in file.readlines():
        new_array=line.split()
        in_data_array.append(new_array)


    in_data_array=np.array(in_data_array)

    in_data_array=in_data_array[:,np.newaxis]

    X1= map(float,in_data_array[:,0,1])
    X2= map(float,in_data_array[:,0,2])

    X=np.c_[X1,X2]
    X=np.array(X)

    Y= map(float,in_data_array[:,0,0])
    Y=np.array(Y)

    return(X,Y)

def get_vsall(Y,value):

    #print Y

    #print (Y==value)

    Y[[Y!=value]]=-1.0
    Y[[Y==value]]=1.0

    return Y

Xtrain,Ytrain= load_data("DigitsIn")
Xtest,Ytest=   load_data("DigitsOut")

for i in np.arange(0,10,1):

    print i

    plt.subplot(5,2,i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Y=get_vsall(Ytest.copy(),i)
    print Y

    x=Xtest[Y==1][:,0]
    y=Xtest[Y==1][:,1]

    print Ytest

    plt.scatter(x,y,c=Ytest[Y==1])
    plt.title("scatter plot of digit " + str(i))
    plt.xlim(Xtrain[:,0].min(),Xtrain[:,0].max())
    plt.ylim(Xtrain[:,1].min(),Xtrain[:,1].max())
    plt.grid()

print "Runtime=" + str(time.time()-t1)

plt.show()

print "Uptime=" + str(time.time()-t1)