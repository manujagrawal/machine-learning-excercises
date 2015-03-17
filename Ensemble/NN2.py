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




DataX=np.arange(-1,1,0.002)
DataY=map(trans,DataX)
DataY=np.array(DataY)
DataX.resize((DataX.shape[0],1))
DataY.resize((DataY.shape[0],1))


size = len(DataX)

trainratio=0.05

randIndex1= random.sample(xrange(size), int(size*trainratio))
inp1 = DataX[randIndex1].copy()
tar1 = DataY[randIndex1].copy()
net1 = nl.net.newff([[-1, 1]],[5, 1])
error1 = net1.train(inp1, tar1, epochs=500, show=1, goal=0.02)
out1 = net1.sim(inp1)

print "******************* "
randIndex2= random.sample(xrange(size), int(size*trainratio))
inp2 = DataX[randIndex2].copy()
tar2 = DataY[randIndex2].copy()
net2 = nl.net.newff([[-1, 1]],[5, 1])
error2 = net2.train(inp2, tar2, epochs=500, show=1, goal=0.02)
out2 = net2.sim(inp2)



plt.subplot(2,2,2)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(inp1[:,0], out1[:,0],'.')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min(), DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 1")

plt.subplot(2,2,3)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(inp2[:,0], out2[:,0],'.')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min(), DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function 2")


plt.subplot(2,2,1)
plt.plot(DataX[:,0], DataY[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min(), DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Actual Function")

plt.show()