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



def makeEnsembles(raw_data,num_clusters,tr_ratio,nn_ratio,n_epochs=3):

	print "function called"

	kmeans=KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
	kmeans.fit(raw_data[:,1:])
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

	size=len(raw_data)

	# convert complete dataset to compatible pybrain data set

	randIndex = random.sample(xrange(size), int(size*tr_ratio))
	trdt=raw_data[randIndex]
	randIndex = random.sample(xrange(size), int(size*(1-tr_ratio)  ))
	tstdt=raw_data[randIndex]
	
	randIndex1 = random.sample(xrange(len(trdt)), int((len(trdt)*nn_ratio)))
	randIndex2 = random.sample(xrange(len(trdt)), int((len(trdt)*nn_ratio)))
	randIndex3 = random.sample(xrange(len(trdt)), int((len(trdt)*nn_ratio)))
	randIndex4 = random.sample(xrange(len(trdt)), int((len(trdt)*nn_ratio)))
	randIndex5 = random.sample(xrange(len(trdt)), int((len(trdt)*nn_ratio)))

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

	
	# make feed forward neural networks----
	 

	fnn1 = buildNetwork( trdata1.indim, 5 , trdata1.outdim, outclass=SoftmaxLayer )
	trainer1 = BackpropTrainer( fnn1, dataset=trdata1, momentum=0.1, learningrate=0.01 , verbose=False, weightdecay=0.01) 
	

	fnn2 = buildNetwork( trdata2.indim, 14 , trdata2.outdim, outclass=SoftmaxLayer )
	trainer2 = BackpropTrainer( fnn2, dataset=trdata2, momentum=0.1, learningrate=0.01 , verbose=False, weightdecay=0.01)
	
	fnn3 = buildNetwork( trdata3.indim, 4 , trdata3.outdim, outclass=SoftmaxLayer )
	trainer3 = BackpropTrainer( fnn3, dataset=trdata3, momentum=0.1, learningrate=0.01 , verbose=False, weightdecay=0.01)
	
	fnn4 = buildNetwork( trdata4.indim, 4 , trdata4.outdim, outclass=SoftmaxLayer )
	trainer4 = BackpropTrainer( fnn4, dataset=trdata4, momentum=0.1, learningrate=0.01 , verbose=False, weightdecay=0.01)
	
	fnn5 = buildNetwork( trdata5.indim, 10 , trdata5.outdim, outclass=SoftmaxLayer )
	trainer5 = BackpropTrainer( fnn5, dataset=trdata5, momentum=0.1, learningrate=0.01 , verbose=False, weightdecay=0.01)
	
	#-----

	

	# train networks-----

	for i,trainer in enumerate((trainer1,trainer2,trainer3,trainer4,trainer5)):
		
		print "net ", i, " --"
		trainer.trainEpochs (n_epochs)

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

	error_on_clusters=np.c_[error1,error2,error3,error4,error5]

	# compute weights on each cluster

	weights=[]

	for row in error_on_clusters:
		deno= ( (1.0/(row[0]+0.01))+(1.0/(row[1]+0.01))+(1.0/(row[2]+0.01))+(1.0/(row[3]+0.01))+(1.0/(row[4]+0.01))    )
		n1,n2,n3,n4,n5 = (1.0/(row[0]+0.01)),(1.0/(row[1]+0.01)),(1.0/(row[2]+0.01)),(1.0/(row[3]+0.01)),(1.0/(row[4]+0.01))
		weights.append([ n1/deno, n2/deno, n3/deno, n4/deno, n5/deno ])

	weights=np.array(weights)

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
		
	insample_error=np.array( [ insample1[0],insample2[0],insample3[0],insample4[0],insample5[0]] )	

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
		
		
	outsample_error=np.array( [ outsample1[0],outsample2[0],outsample3[0],outsample4[0],outsample5[0]] )	

	return (kmeans,
			compatible_clusters,
			trdata,
			tstdata,
			fnn1,fnn2,fnn3,fnn4,fnn5,
			#{'fnn':fnn1,'trainer':trainer1,'trdata':trdata1},
			#{'fnn':fnn2,'trainer':trainer2,'trdata':trdata2},
			#{'fnn':fnn3,'trainer':trainer3,'trdata':trdata3},
			#{'fnn':fnn4,'trainer':trainer4,'trdata':trdata4},
			#{'fnn':fnn5,'trainer':trainer5,'trdata':trdata5},
			weights,
			insample_error,
			outsample_error)



def cal_ensemble_error(raw_data,num_clusters,tr_ratio,epochs):

	my_raw_data=raw_data
	
	kmeans,compatible_clusters,train_data,tst_data,est1,est2,est3,est4,est5,weights,insample_error,outsample_error = makeEnsembles(raw_data=my_raw_data, num_clusters= num_clusters, tr_ratio=tr_ratio, nn_ratio=0.30 ,n_epochs=epochs)

	tr_out=[]
	tst_out=[]

	tr_out1=est1.activateOnDataset(train_data)
	tr_out2=est2.activateOnDataset(train_data)
	tr_out3=est3.activateOnDataset(train_data)
	tr_out4=est4.activateOnDataset(train_data)
	tr_out5=est5.activateOnDataset(train_data)

	tst_out1=est1.activateOnDataset(tst_data)
	tst_out2=est2.activateOnDataset(tst_data)
	tst_out3=est3.activateOnDataset(tst_data)
	tst_out4=est4.activateOnDataset(tst_data)
	tst_out5=est5.activateOnDataset(tst_data)


	j=0
	
	for row in train_data['input']:
		i=kmeans.predict(row)

		o1=weights[i,0]*np.array(tr_out1[j])
		o2=weights[i,1]*np.array(tr_out1[j])
		o3=weights[i,2]*np.array(tr_out1[j])
		o4=weights[i,3]*np.array(tr_out1[j])
		o5=weights[i,4]*np.array(tr_out1[j])
		
		tr_out.append(o1+o2+o3+o4+o5)

		j+=1
	
	tr_out=np.array(tr_out)	
	out1=np.argmax( np.array(train_data['target']),axis=1 )
	
	en_tr_out=np.argmax(tr_out,axis=1)
	in_sample_error= ( float(np.array(en_tr_out!=out1).sum())/len(en_tr_out) )*100
	
	j=0
		
	for row in tst_data['input']:
		i=kmeans.predict(row)
		
		o1=weights[i,0]*np.array(tst_out1[j])
		o2=weights[i,1]*np.array(tst_out1[j])
		o3=weights[i,2]*np.array(tst_out1[j])
		o4=weights[i,3]*np.array(tst_out1[j])
		o5=weights[i,4]*np.array(tst_out1[j])
		
		tst_out.append(o1+o2+o3+o4+o5)

		j+=1

	tst_out=np.array(tst_out)
	en_tst_out=  np.argmax(tst_out,axis=1)
	out=np.argmax( np.array(tst_data['target']), axis=1)
	
	
	out_sample_error= ( float(np.array(en_tst_out!=out).sum())/len(en_tst_out) )*100

	avg_insample_error = np.average(insample_error)
	avg_outsample_error= np.average(outsample_error)
	
	
	return (avg_insample_error,avg_outsample_error,in_sample_error,out_sample_error)


data2,target=make_blobs(n_samples=2000,n_features=7,centers=4,shuffle=True,cluster_std=3)
data2=np.array(data2)
target=np.array(target)
my_raw_data=np.concatenate((target.reshape(target.shape[0],1),data2),axis=1)

#n_clusters=[ int(4500.0/10),int(4500.0/30)  ]
#tr_ratio  =[0.1,0.8]  


n_clusters=[ int(4500.0/30),int(4500.0/40),int(4500.0/50),int(4500.0/60),int(4500.0/70),int(4500.0/80),int(4500.0/90),int(4500.0/100)  ]
tr_ratio  =[0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8]  

avg_In_1=[]
avg_Out_1=[]
In_1=[]
Out_1=[]

avg_In_2=[]
avg_Out_2=[]
In_2=[]
Out_2=[]

for num_clus in n_clusters:
	a_in,a_out,In,out = cal_ensemble_error(raw_data=my_raw_data,num_clusters=num_clus,tr_ratio=0.5,epochs=10)
	avg_In_1.append(a_in)
	avg_Out_1.append(a_out)
	In_1.append(In)
	Out_1.append(out)

now_num_clus= n_clusters[np.argmax(np.array(avg_Out_1))]

for trRatio in tr_ratio:
	a_in,a_out,In,out = cal_ensemble_error(raw_data=my_raw_data,num_clusters=now_num_clus,tr_ratio=trRatio,epochs=10)
	avg_In_2.append(a_in)
	avg_Out_2.append(a_out)
	In_2.append(In)
	Out_2.append(out)

print avg_In_1
print avg_Out_1
print In_1
print Out_1

print avg_In_2
print avg_Out_2
print In_2
print Out_2


plt.subplot(1,2,1)
plt.plot(n_clusters, avg_In_1, 'r--',color='r', label="average insample error of individual networks" ,linewidth=2.0)
plt.plot(n_clusters, avg_Out_1,'r--',color='k', label="average outsample error of individual networks",linewidth=2.0 )
plt.plot(n_clusters, In_1, label="insample error of ensemble networks",linewidth=2.0 )
plt.plot(n_clusters, Out_1, label="outsample error of ensemble networks",linewidth=2.0 )
plt.xlim(np.array(n_clusters).min(), np.array(n_clusters).max())
plt.ylim(0, 60)
plt.legend()
plt.xticks(( np.arange(np.array(n_clusters).min(), np.array(n_clusters).max(),15) ))
plt.yticks((np.arange(0,30,2)))
plt.title("Training Ratio (= 0.5) vs Number of Clusters ")


plt.subplot(1,2,2)
plt.plot(tr_ratio, avg_In_2,'r--',color='r', label="average insample error of individual networks",linewidth=2.0 )
plt.plot(tr_ratio, avg_Out_2,'r--',color='k',label="average outsample error of individual networks" ,linewidth=2.0)
plt.plot(tr_ratio, In_2, label="insample error of ensemble networks" )
plt.plot(tr_ratio, Out_2, label="outsample error of ensemble networks" )
plt.xlim(np.array(tr_ratio).min(), np.array(tr_ratio).max()+0.02)
plt.ylim(0, 60)
plt.legend()
plt.xticks((tr_ratio))
plt.yticks((np.arange(0,70,10)))
plt.title( str(now_num_clus) + " Clusters vs Training Ratio ")

print "Finished in ", time(), " Seconds"

plt.show()

