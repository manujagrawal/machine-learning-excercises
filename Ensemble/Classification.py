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




