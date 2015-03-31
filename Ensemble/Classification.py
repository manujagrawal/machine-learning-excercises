import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn.datasets import make_blobs

data2,target=make_blobs(n_samples=4500,n_features=7,centers=4,shuffle=True,cluster_std=5.0)

data2=np.array(data2)
target=np.array(target)

print target

#data = scale(data)
raw_data=np.concatenate((target.reshape(target.shape[0],1),data2),axis=1)


#print data.shape
print data2.shape
print target.shape
print raw_data.shape

print "data loaded"


# make clusters -----------
    
kmeans=KMeans(init='k-means++', n_clusters=len(data2)/10, n_init=20)
kmeans.fit(data2)
centroids = kmeans.cluster_centers_

print "number of clusters=", len(centroids)

clusters=[]

for i in xrange(len(centroids)):
	cluster=[]
	clusters.append(cluster) 

for row in raw_data:
	for i in  kmeans.predict(row[1:]):
		clusters[i].append(row)

# ---------

print "clusters made"


size=len(data2)

# convert complete dataset to compatible pybrain data set

randIndex = random.sample(xrange(size), int(size*0.30))
trdt=raw_data[randIndex]
print len(trdt)
randIndex = random.sample(xrange(size), int(size*0.70))
tstdt=raw_data[randIndex]
print len(tstdt)

randIndex1 = random.sample(xrange(len(trdt)), int(size*0.10))
randIndex2 = random.sample(xrange(len(trdt)), int(size*0.10))
randIndex3 = random.sample(xrange(len(trdt)), int(size*0.10))
randIndex4 = random.sample(xrange(len(trdt)), int(size*0.10))
randIndex5 = random.sample(xrange(len(trdt)), int(size*0.10))

trdt1=trdt[randIndex1]
trdt2=trdt[randIndex2]
trdt3=trdt[randIndex3]
trdt4=trdt[randIndex4]
trdt5=trdt[randIndex5]

trdata = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt)):
	trdata.addSample(trdt[i][1:], [trdt[i][0]])
trdata._convertToOneOfMany(bounds=[0, 1])                  #---


tstdata = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(tstdt)):
	tstdata.addSample(tstdt[i][1:], [tstdt[i][0]])
tstdata._convertToOneOfMany(bounds=[0, 1])                   #---

trdata1 = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt1)):
	trdata1.addSample(trdt1[i][1:], [trdt1[i][0]])
trdata1._convertToOneOfMany(bounds=[0, 1])                      #---

trdata2 = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt2)):
	trdata2.addSample(trdt2[i][1:], [trdt2[i][0]])
trdata2._convertToOneOfMany(bounds=[0, 1])                      #---

trdata3 = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt3)):
	trdata3.addSample(trdt3[i][1:], [trdt3[i][0]])
trdata3._convertToOneOfMany(bounds=[0, 1])                      #---

trdata4 = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt4)):
	trdata4.addSample(trdt4[i][1:], [trdt4[i][0]])
trdata4._convertToOneOfMany(bounds=[0, 1])                      #---

trdata5 = ClassificationDataSet(7,1, nb_classes=4)
for i in xrange(len(trdt5)):
	trdata5.addSample(trdt5[i][1:], [trdt5[i][0]])
trdata5._convertToOneOfMany(bounds=[0, 1])                      #---

#---------------------------------
print "compatible data set made "


# convert cluster data set to compatible pybrain data set
compatible_clusters=[]

for cluster in clusters:
	ds = ClassificationDataSet(7,1, nb_classes=4)
	for i in xrange(len(cluster)):
		a=cluster[i][1:]
		b=[cluster[i][0]]
		ds.appendLinked(a,b)
	ds._convertToOneOfMany(bounds=[0, 1])

	compatible_clusters.append(ds)
#-------------------

print "compatible clusters made"

# make feed forward neural networks----
 

fnn1 = buildNetwork( trdata1.indim, 5 , trdata1.outdim, outclass=SoftmaxLayer )
trainer1 = BackpropTrainer( fnn1, dataset=trdata1, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 
print trdata1.calculateStatistics()

fnn2 = buildNetwork( trdata2.indim, 2 , trdata2.outdim, outclass=SoftmaxLayer )
trainer2 = BackpropTrainer( fnn2, dataset=trdata2, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
print trdata2.calculateStatistics()

fnn3 = buildNetwork( trdata3.indim, 4 , trdata3.outdim, outclass=SoftmaxLayer )
trainer3 = BackpropTrainer( fnn3, dataset=trdata3, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
print trdata3.calculateStatistics()

fnn4 = buildNetwork( trdata4.indim, 4 , trdata4.outdim, outclass=SoftmaxLayer )
trainer4 = BackpropTrainer( fnn4, dataset=trdata4, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
print trdata4.calculateStatistics()

fnn5 = buildNetwork( trdata5.indim, 10 , trdata5.outdim, outclass=SoftmaxLayer )
trainer5 = BackpropTrainer( fnn5, dataset=trdata5, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
print trdata5.calculateStatistics()

#-----

print "neural networks made"


# train networks-----

for i,trainer in enumerate((trainer1,trainer2,trainer3,trainer4,trainer5)):
	
	print "training network ", i, " ------------------------------------"
	trainer.trainEpochs (2)

#-----------

error1=[]
error2=[]
error3=[]
error4=[]
error5=[]

for error,fnn in [ [error1,fnn1],[error2,fnn2],[error3,fnn3],[error4,fnn4],[error5,fnn5]]:
	for ds in compatible_clusters:
		out1 = np.array( fnn.activateOnDataset(ds).argmax(axis=1) ) 
		out = np.array(  ds['target'].argmax(axis=1))
		P_error = ( float(np.array(out1!=out).sum())/len(out) )*100

		error.append( P_error ) 


for error in [error1,error2,error3,error4,error5]:
	error=np.array(error)

error=np.c_[error1,error2,error3,error4,error5]

# compute weights on each cluster

print "len error =", len(error)

weights=[]

for row in error:
	deno= ( (1.0/(row[0]+0.1))+(1.0/(row[1]+0.1))+(1.0/(row[2]+0.1))+(1.0/(row[3]+0.1))+(1.0/(row[4]+0.1))    )
	n1,n2,n3,n4,n5 = (1.0/(row[0]+0.1)),(1.0/(row[1]+0.1)),(1.0/(row[2]+0.1)),(1.0/(row[3]+0.1)),(1.0/(row[4]+0.1))
	weights.append([ n1/deno, n2/deno, n3/deno, n4/deno, n5/deno ])

weights=np.array(weights)

for row in weights:
	print row, row.sum()



insample1=[]
insample2=[]
insample3=[]
insample4=[]
insample5=[]
i=1
#computing insample error (trdata) for each of the networks
for  insample,fnn in [ [insample1,fnn1],[insample2,fnn2],[insample3,fnn3],[insample4,fnn4],[insample5,fnn5] ]:
	out1 = np.array( fnn.activateOnDataset(trdata).argmax(axis=1) ) 
	out = np.array(  trdata['target'].argmax(axis=1))
	P_error = ( float(np.array(out1!=out).sum())/len(out) )*100
	insample.append( P_error )
	print "insample error for ", i, '=', insample
	i+=1


#-----------

#computing outsample error tstdata for each of the networks


outsample1=[]
outsample2=[]
outsample3=[]
outsample4=[]
outsample5=[]
i=1

for outsample,fnn in  [ [outsample1,fnn1],[outsample2,fnn2],[outsample3,fnn3],[outsample4,fnn4],[outsample5,fnn5] ] :
	out1 = np.array( fnn.activateOnDataset(tstdata).argmax(axis=1) ) 
	out = np.array(  tstdata['target'].argmax(axis=1))
	P_error = ( float(np.array(out1!=out).sum())/len(out) )*100
	outsample.append( P_error )
	print "outsample error for ", i, '=', outsample
	i+=1

#----------








"""

def makeNN():

    n1=FeedForwardNetwork()

    inLayer=SigmoidLayer(14)
    h1=SigmoidLayer(14)
    h2=SigmoidLayer(14)
    outLayer=LinearLayer(2)

    n1.addInputModule(inLayer)
    n1.addModule(h1)
    n1.addModule(h2)
    n1.addOutputModule(outLayer)

    in_h1=FullConnection(inLayer,h1)
    h1_h2=FullConnection(h1,h2)
    h2_out=FullConnection(h2,outLayer)

    n1.addConnection(in_h1)
    n1.addConnection(h1_h2)
    n1.addConnection(h2_out)

    n1.sortModules()

    return (n1)
"""




