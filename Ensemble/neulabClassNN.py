__author__ = 'manuj'

import neurolab as nl
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def makeRandomNumbers(a,b,n):
    numbers=[]
    for i in xrange(n):
        numbers.append(random.uniform(a,b))
    return numbers
def trans(x):
    return (float(1)/math.exp(10*math.pow(x,2)))

class NN:

    def __init__(self,nn,dataX,dataY): # dataX in form [ [a,b],[b,c],[c,g] ] ////  dataY in form [ [t],[t],[t] ]
        self.X=np.array(dataX)
        self.Y=np.array(dataY)
        self.net=nn

    def train(self,nepochs=450,nshow=1):
        print "**"

        error = net.train(self.X, self.Y, epochs=nepochs, show=nshow, goal=0.02)

        print "**** finished training this network ***"
        self.out=self.net.sim(self.X)
        self.out=np.array(self.out)

        myarray= np.array(self.out[:,0]-self.Y[:,0])
        print self.out[:,0].shape
        print self.Y[:,0].shape
        print myarray.shape

        self.error=[math.pow(a, 2) for a in (myarray)]
        self.error=np.array(self.error)
        self.error.ravel()

    def predict(self,dataX):  # dataX in form [ [a,b],[b,c],[c,g] ]

        out=np.array( self.net.sim(dataX) )
        out.ravel()
        return out.copy()  # returns np.array in form [a,b,c,d,,,,,]



class ensembleNN:

    def __init__(self,dataX,dataY):       # dataX in form [ [a,b],[b,c],[c,g] ] ////  dataY in form [ [t],[t],[t] ]
        self.dataX=np.array(dataX)
        self.dataY=np.array(dataY)
        self.size= len(self.dataX)
        self.net=[]
        self.num=0

    def addNN(self,nn,trainratio):
        randIndex= random.sample(xrange(self.size), int(self.size*trainratio))
        net=NN(nn.copy(), dataX=self.dataX[randIndex].copy(), dataY=self.dataY[randIndex].copy())
        self.net.append(net)
        self.num=self.num+1

    def getNNcount(self):
        return self.num

    def trainAll(self,nepochs=450,nshow=1):
        print "-------------------------------------------"

        for i in xrange(self.num):
            print "training network number:", i+1
            self.net[i].train(nepochs,nshow)

    def trainPnetwork(self,p,nepochs=450,nshow=5,):
        print "-------------------------------------------"
        print "training network number:", p+1

        self.net[p+1].train(nepochs,nshow)

    def displyAnalysis(self):

        for i in xrange(self.num):

            plt.subplot(int (self.num+1/2), 2, i+1)
            plt.plot(self.dataX[:,0],self.dataY[:,0])

            #andSampleX = np.array(self.dataX[ random.sample(xrange(self.size), int(self.size*0.8)) ])  # of form [ [],[],[],,,, ]
            #predicted= np.array(self.net[i].predict(randSampleX ) )
            #plt.plot(  randSampleX.ravel(),predicted.ravel(),'.' )

            plt.plot(  self.net[i].X[:,0] , self.net[i].out[:,0],'.' )

            plt.xlim(self.dataX[:,0].min()-0.1, self.dataX[:,0].max()+0.1)
            plt.ylim(self.dataY[:,0].min()-0.1, self.dataY.max()+0.1)
            plt.xticks((np.arange(self.dataX[:,0].min()-0.1, self.dataX[:,0].max()+0.1,0.5)))
            plt.yticks((np.arange(int(self.dataY[:,0].min()), int(self.dataY[:,0].max()),0.2)))
            plt.title("Result of Network " + str(i+1))

        plt.show()



DataX=np.arange(-1,1,0.002)
DataY=map(trans,DataX)
DataY=np.array(DataY)
DataX.resize((DataX.shape[0],1))
DataY.resize((DataY.shape[0],1))


ensemble=ensembleNN(dataX=DataX,dataY=DataY)

for i in xrange(5):
    net=nl.net.newff([[-1, 1]],[4,4, 1])
    ensemble.addNN(nn=net.copy(), trainratio=0.05)

ensemble.trainAll()
ensemble.displyAnalysis()


"""
plt.subplot(1,2,1)
plt.plot(fullDataX[:,0],fullDataY[:,0])
plt.plot(fullDataX[:,0],out)
plt.xlim(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1)
plt.ylim(fullDataY.min(), fullDataY.max()+0.1)
plt.xticks((np.arange(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(fullDataY.min()), int(fullDataY.max()),0.2)))
plt.title("Actual function")

plt.subplot(1,2,2)
plt.plot(train1X[:,0], (net.sim(train1X))[:,0],'.' )
plt.xlim(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1)
plt.ylim(fullDataY.min(), fullDataY.max()+0.1)
plt.xticks((np.arange(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(fullDataY.min()), int(fullDataY.max()),0.2)))
plt.title("Estimated function")

plt.show()

"""
