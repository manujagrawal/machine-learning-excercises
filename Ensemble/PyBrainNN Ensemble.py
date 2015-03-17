from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from copy import deepcopy
import random
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold

t1=time.time()

def makeNN():

    n1=FeedForwardNetwork()

    inLayer=SigmoidLayer(1)
    h1=SigmoidLayer(20)
    h2=SigmoidLayer(20)
    h3=SigmoidLayer(20)
    h4=SigmoidLayer(20)
    outLayer=LinearLayer(1)

    n1.addInputModule(inLayer)
    n1.addModule(h1)
    n1.addModule(h2)
    n1.addModule(h3)
    n1.addModule(h4)
    n1.addOutputModule(outLayer)

    in_h1=FullConnection(inLayer,h1)
    h1_h2=FullConnection(h1,h2)
    h2_h3=FullConnection(h2,h3)
    h3_h4=FullConnection(h3,h4)
    h4_out=FullConnection(h4,outLayer)

    n1.addConnection(in_h1)
    n1.addConnection(h1_h2)
    n1.addConnection(h2_h3)
    n1.addConnection(h3_h4)
    n1.addConnection(h4_out)

    n1.sortModules()

    return deepcopy(n1)
def makeNNlist(num):

    nnList=[]

    for i in xrange(num):
        nnList.append(makeNN())

    return nnList
def makeRandomNumbers(a,b,n):
    numbers=[]
    for i in xrange(n):
        numbers.append(random.uniform(a,b))
    return numbers
def trans(x):
    return (float(100)/math.exp(10*math.pow(x,2)))

def convertDataNN(data, values):

    fulldata = SupervisedDataSet(data.shape[1], 1)

    for d, v in zip(data, values):

        fulldata.addSample(d, v)

    return fulldata

fullDataX=np.arange(-1,1,0.002)
fullDataY=map(trans,fullDataX)
fullDataY=np.array(fullDataY)
fullDataX.resize((fullDataX.shape[0],1))
fullDataY.resize((fullDataY.shape[0],1))


indextrain1=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))
indextrain2=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))
indextrain3=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))
indextrain4=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))
indextrain5=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))
indextrain6=np.array(random.sample(xrange(fullDataX[:,0].shape[0]), 600))


train1X=fullDataX[indextrain1]
train1Y=fullDataY[indextrain1]

train2X=fullDataX[indextrain2]
train2Y=fullDataY[indextrain2]

train3X=fullDataX[indextrain3]
train3Y=fullDataY[indextrain3]

train4X=fullDataX[indextrain4]
train4Y=fullDataY[indextrain4]

train5X=fullDataX[indextrain5]
train5Y=fullDataY[indextrain5]

train6X=fullDataX[indextrain6]
train6Y=fullDataY[indextrain6]


train1=convertDataNN(train1X,train1Y)
train2=convertDataNN(train2X,train2Y)
train3=convertDataNN(train3X,train3Y)
train4=convertDataNN(train4X,train4Y)
train5=convertDataNN(train5X,train5Y)
train6=convertDataNN(train6X,train6Y)



NN=makeNNlist(6)

NN[0].activate((fullDataX[0]))
NN[1].activate((fullDataX[100]))
NN[2].activate((fullDataX[400]))
NN[3].activate((fullDataX[600]))
NN[4].activate((fullDataX[800]))
NN[5].activate((fullDataX[940]))

trainer1 = BackpropTrainer(NN[0], dataset=train1, momentum=0.1, verbose=True, weightdecay=0.001)
trainer2 = BackpropTrainer(NN[1], dataset=train2, momentum=0.1, verbose=True, weightdecay=0.001)
trainer3 = BackpropTrainer(NN[2], dataset=train3, momentum=0.1, verbose=True, weightdecay=0.001)
trainer4 = BackpropTrainer(NN[3], dataset=train4, momentum=0.1, verbose=True, weightdecay=0.001)
trainer5 = BackpropTrainer(NN[4], dataset=train5, momentum=0.1, verbose=True, weightdecay=0.001)
trainer6 = BackpropTrainer(NN[5], dataset=train6, momentum=0.1, verbose=True, weightdecay=0.001)

trainer1.trainUntilConvergence()
for i in range(1):
    trainer1.trainEpochs(1)
    trainer2.trainEpochs(1)
    trainer3.trainEpochs(1)
    trainer4.trainEpochs(1)
    trainer5.trainEpochs(1)
    trainer6.trainEpochs(1)

print (NN[0].activate(([-0.9])))

print "Runtime=" + str(time.time()-t1)

plt.plot(fullDataX[:,0],fullDataY)
plt.xlim(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1)
plt.ylim(fullDataY.min(), fullDataY.max()+0.1)
plt.xticks((np.arange(fullDataX[:,0].min()-0.1, fullDataX[:,0].max()+0.1,0.5)))
plt.yticks((np.arange(int(fullDataY.min()), int(fullDataY.max()+6),5)))
#plt.show()




