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

data=[]
target=[]

with open('train.csv') as csvfile:
		reader=csv.reader(csvfile,delimiter=',')
		reader.next()
		for row in reader:
			data.append([ float(i.replace(',','')) for i in row[1:] ] )
			target.append(float(row[0]))


data=np.array(data)
target=np.array(target)
data = scale(data)
raw_data=np.concatenate((target.reshape(target.shape[0],1),data),axis=1)

n_samples, n_features = data.shape
n_targets = 2

print "data loaded"


# make clusters -----------
    
kmeans=KMeans(init='k-means++', n_clusters=n_samples/20, n_init=10)
kmeans.fit(data)
centroids = kmeans.cluster_centers_

clusters=[]

for i in xrange(224):
	cluster=[]
	clusters.append(cluster) 

for row in raw_data:
	for i in  kmeans.predict(row[1:]):
		clusters[i].append(row)

# ---------

print "clusters made"


size=len(data)

# convert complete dataset to compatible pybrain data set

randIndex = random.sample(xrange(size), int(size*0.65))
trdt=raw_data[randIndex]
tstdt=raw_data[randIndex]

randIndex1 = random.sample(xrange(size), int(size*0.50))
randIndex2 = random.sample(xrange(size), int(size*0.50))
randIndex3 = random.sample(xrange(size), int(size*0.50))
randIndex4 = random.sample(xrange(size), int(size*0.50))
randIndex5 = random.sample(xrange(size), int(size*0.50))

trdt1=raw_data[randIndex1]
trdt2=raw_data[randIndex2]
trdt3=raw_data[randIndex3]
trdt4=raw_data[randIndex4]
trdt5=raw_data[randIndex5]

trdata = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt)):
	trdata.addSample(trdt[i][1:], [trdt[i][0]])
trdata._convertToOneOfMany(bounds=[0, 1])                  #---


tstdata = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(tstdt)):
	tstdata.addSample(tstdt[i][1:], [tstdt[i][0]])
tstdata._convertToOneOfMany(bounds=[0, 1])                   #---

trdata1 = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt1)):
	trdata1.addSample(trdt1[i][1:], [trdt1[i][0]])
trdata1._convertToOneOfMany(bounds=[0, 1])                      #---

trdata2 = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt2)):
	trdata2.addSample(trdt2[i][1:], [trdt2[i][0]])
trdata2._convertToOneOfMany(bounds=[0, 1])                      #---

trdata3 = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt3)):
	trdata3.addSample(trdt3[i][1:], [trdt3[i][0]])
trdata3._convertToOneOfMany(bounds=[0, 1])                      #---

trdata4 = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt4)):
	trdata4.addSample(trdt4[i][1:], [trdt4[i][0]])
trdata4._convertToOneOfMany(bounds=[0, 1])                      #---

trdata5 = ClassificationDataSet(14,2, nb_classes=2)
for i in xrange(len(trdt5)):
	trdata5.addSample(trdt5[i][1:], [trdt5[i][0]])
trdata5._convertToOneOfMany(bounds=[0, 1])                      #---

#---------------------------------
print "compatible data set made "


# convert cluster data set to compatible pybrain data set
compatible_clusters=[]

for cluster in clusters:
	ds = ClassificationDataSet(14,2, nb_classes=2)
	for i in xrange(len(cluster)):
		a=cluster[i][1:]
		b=[cluster[i][0]]
		ds.appendLinked(a,b)
	ds._convertToOneOfMany(bounds=[0, 1])

	compatible_clusters.append(ds)
#-------------------

print "compatible clusters made"

# make feed forward neural networks----
fnn1 = buildNetwork( trdata1.indim, 7 , trdata1.outdim, outclass=SoftmaxLayer )
trainer1 = BackpropTrainer( fnn1, dataset=trdata1, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 


fnn2 = buildNetwork( trdata2.indim, 7 , trdata2.outdim, outclass=SoftmaxLayer )
trainer2 = BackpropTrainer( fnn2, dataset=trdata2, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)


fnn3 = buildNetwork( trdata3.indim, 7 , trdata3.outdim, outclass=SoftmaxLayer )
trainer3 = BackpropTrainer( fnn3, dataset=trdata3, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)


fnn4 = buildNetwork( trdata4.indim, 7 , trdata4.outdim, outclass=SoftmaxLayer )
trainer4 = BackpropTrainer( fnn4, dataset=trdata4, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)


fnn5 = buildNetwork( trdata5.indim, 7 , trdata5.outdim, outclass=SoftmaxLayer )
trainer5 = BackpropTrainer( fnn5, dataset=trdata5, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)

#-----

print "neural networks made"


# train networks-----

print "training network 1------------------------------------"
trainer1.trainEpochs (50)

print "training network 2------------------------------------"
trainer2.trainEpochs (50)


print "training network 3------------------------------------"
trainer3.trainEpochs (50)


print "training network 4------------------------------------"
trainer4.trainEpochs (50)


print "training network 5------------------------------------"
trainer5.trainEpochs (50)
#-----------






"""
with open('test.csv') as csvfile:
		reader=csv.reader(csvfile,delimiter=',')
		reader.next()
		for row in reader:
			print kmeans.predict([ float(i.replace(',','')) for i in row ])

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




