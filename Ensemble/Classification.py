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


with open('train.csv') as csvfile:
		reader=csv.reader(csvfile,delimiter=',')
		reader.next()
		for row in reader:
			data.append([ float(i.replace(',','')) for i in row[1:] ] )
			target.append(float(row[0]))


data=np.array(data)
target=np.array(target)
#data = scale(data)

raw_data=np.concatenate((target.reshape(target.shape[0],1),data),axis=1)

print data.shape
print target.shape
print raw_data.shape

n_samples, n_features = data.shape
n_targets = 2

# make clusters -----------
    
kmeans=KMeans(init='k-means++', n_clusters=n_samples/20, n_init=10)
kmeans.fit(data)
centroids = kmeans.cluster_centers_

print centroids.shape

clusters=[]

for i in xrange(224):
	cluster=[]
	clusters.append(cluster) 

for row in raw_data:
	for i in  kmeans.predict(row[1:]):
		clusters[i].append(row)

# clusters made -----------

"""
with open('test.csv') as csvfile:
		reader=csv.reader(csvfile,delimiter=',')
		reader.next()
		for row in reader:
			print kmeans.predict([ float(i.replace(',','')) for i in row ])

"""			
"""
inLayer=[ 	[ data[:,0].min(),data[:,0].max() ],
			[ data[:,1].min(),data[:,1].max() ],
			[ data[:,2].min(),data[:,2].max() ],
			[ data[:,3].min(),data[:,3].max() ],
			[ data[:,4].min(),data[:,4].max() ],
			[ data[:,5].min(),data[:,5].max() ],
			[ data[:,6].min(),data[:,6].max() ],
			[ data[:,7].min(),data[:,7].max() ],
			[ data[:,8].min(),data[:,8].max() ],
			[ data[:,9].min(),data[:,9].max() ],
			[ data[:,10].min(),data[:,10].max() ],
			[ data[:,11].min(),data[:,11].max() ],
			[ data[:,12].min(),data[:,12].max() ],
			[ data[:,13].min(),data[:,13].max() ] ]   

outLayer=[14, 2]
"""
size=len(data)
tratio=0.6

modified_target=[]

for row in target:
	newrow=[0,0]
	newrow[int(row)]=1
	modified_target.append(newrow)

modified_target=np.array(modified_target)

DS = ClassificationDataSet(14, nb_classes=2)

for i in xrange(len(raw_data)):
	DS.appendLinked(raw_data[i][1:], [raw_data[i][0]])

DS._convertToOneOfMany(bounds=[0, 1])

fnn = buildNetwork( DS.indim, 7 , DS.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=DS, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

print 'training started'

trainer.trainEpochs (50)


#print "training started "

""""
nn=makeNN()
nn.activate((data[0]))
trainer1 = BackpropTrainer(nn, dataset=train1, momentum=0.1, verbose=True, weightdecay=0.001)
"""
#randIndex= random.sample(xrange(size), int(size*tratio))
#net = nl.net.newff(inLayer,outLayer)
#error = net.train(data[randIndex], modified_target[randIndex], epochs=600, show=1, goal=0.02)




#print "training completed"

#print error 









"""
def trainNN ():
    randIndex= random.sample(xrange(size), int(size*tratio))
    net = nl.net.newff(inLayer,outLayer)
    error = net.train(data[randIndex], modified_target[randIndex], epochs=600, show=1, goal=0.02)

    out = net.sim(inp)
    fout= net.sim(X)
    error=np.array(error)

    myarray= np.array( np.array(fout[:,0])-np.array(Y[:,0]) )
    msqerror=[]

    for a in myarray:
        #b=math.pow(a, 2)
        b=abs(a)

        msqerror.append(b)
    msqerror=np.array(msqerror)
    #msqerror=[math.pow(a, 2) for a in (myarray)]
    #msqerror=np.array(error)
    #msqerror.ravel()

    return (net,inp,tar,out,fout,msqerror,error.min())

"""

