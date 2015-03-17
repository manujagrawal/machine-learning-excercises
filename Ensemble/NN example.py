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

trainratio=0.1

randIndex= random.sample(xrange(size), int(size*trainratio))
inp = DataX[randIndex]
tar = DataY[randIndex]


net = nl.net.newff([[-1, 1]],[5, 1])

# Train network
error = net.train(inp, tar, epochs=500, show=1, goal=0.02)

# Simulate network
out = net.sim(inp)

plt.subplot(1,2,1)
plt.plot(DataX[:,0], DataY[:,0] )
plt.plot(inp[:,0], out[:,0],'.')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min(), DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Estimated function")

plt.subplot(1,2,2)
plt.plot(DataX[:,0], DataY[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min(), DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(DataY.min()), int(DataY.max()),0.2)))
plt.title("Actual Function")

plt.show()