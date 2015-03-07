__author__ = 'manuj'


__author__ = 'manuj'
import neurolab as nl
import numpy as np
import random
import matplotlib.pyplot as plt
import math


def makeRandomNumbers(a,b,n):
    numbers=[]
    for i in xrange(n):
        numbers.append(random.uniform(a,b))
    return numbers

def trans(x):
    return (float(1)/math.exp(10*math.pow(x,2)))


def trainNN (X,Y,tratio,inLayer,outLayer):
    size = len(X)
    randIndex= random.sample(xrange(size), int(size*tratio))
    inp = X[randIndex]
    tar = Y[randIndex]
    net = nl.net.newff(inLayer,outLayer)
    error = net.train(inp, tar, epochs=500, show=1, goal=0.02)
    out = net.sim(inp)
    fout= net.sim(X)

    myarray= np.array( np.array(fout[:,0])-np.array(Y[:,0]) )
    msqerror=[]

    for a in myarray:
        b=math.pow(a, 2)
        msqerror.append(b)
    msqerror=np.array(msqerror)
    #msqerror=[math.pow(a, 2) for a in (myarray)]
    #msqerror=np.array(error)
    #msqerror.ravel()

    return (net,inp,tar,out,fout,msqerror)


DataX=np.arange(-1,1,0.002)
DataY=map(trans,DataX)
DataY=np.array(DataY)
DataX.resize((DataX.shape[0],1))
DataY.resize((DataY.shape[0],1))

trainR=0.030

net1,inp1,tar1,out1,allOut1,msqerror1 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])
#allOut1=net1.sim(DataX)

print "******************* "

net2,inp2,tar2,out2,allOut2,msqerror2 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])
#allOut2=net2.sim(DataX)

print "******************* "

net3,inp3,tar3,out3,allOut3,msqerror3 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])
#allOut3=net3.sim(DataX)

print "******************* "

net4,inp4,tar4,out4,allOut4,msqerror4 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])
#allOut4=net4.sim(DataX)

print "******************* "

net5,inp5,tar5,out5,allOut5,msqerror5 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])

print "******************* "

net6,inp6,tar6,out6,allOut6,msqerror6 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[-1, 1]], outLayer=[5, 1])


plt.subplot(2,4,2)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut1[:,0],)
plt.plot(DataX[:,0], msqerror1,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 1")

plt.subplot(2,4,3)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut2[:,0])
plt.plot(DataX[:,0], msqerror2,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 2")


plt.subplot(2,4,4)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut3[:,0])
plt.plot(DataX[:,0], msqerror3,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 3")

plt.subplot(2,4,5)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut4[:,0])
plt.plot(DataX[:,0], msqerror4,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 4")

plt.subplot(2,4,6)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut5[:,0])
plt.plot(DataX[:,0], msqerror5,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 5")

plt.subplot(2,4,7)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(DataX[:,0], allOut6[:,0])
plt.plot(DataX[:,0], msqerror6,'-')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 6")



plt.subplot(2,4,1)
plt.plot(DataX[:,0], DataY[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Actual Function")

plt.show()