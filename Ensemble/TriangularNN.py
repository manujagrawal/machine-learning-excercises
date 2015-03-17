__author__ = 'manuj'
__author__ = 'manuj'


__author__ = 'manuj'
import neurolab as nl
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time

#new comment



ti=time.time()

def makeRandomNumbers(a,b,n):
    numbers=[]
    for i in xrange(n):
        numbers.append(random.uniform(a,b))
    return numbers

def trans(x):
    return (2*(x-int(x)) -1 )

#def trans(x):
#    return math.sin(3.14*x)


def trainNN (X,Y,tratio,inLayer,outLayer):
    size = len(X)
    randIndex= random.sample(xrange(size), int(size*tratio))
    inp = X[randIndex]
    tar = Y[randIndex]
    net = nl.net.newff(inLayer,outLayer)
    error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
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


DataX=np.arange(0,2,0.002)
DataY=map(trans,DataX)
DataY=np.array(DataY)
DataX.resize((DataX.shape[0],1))
DataY.resize((DataY.shape[0],1))

trainR=0.03

net1,inp1,tar1,out1,allOut1,msqerror1,error1 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5,5 ,1])
#allOut1=net1.sim(DataX)

print "******************* "

net2,inp2,tar2,out2,allOut2,msqerror2,error2 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5, 5,1])
#allOut2=net2.sim(DataX)

print "******************* "

net3,inp3,tar3,out3,allOut3,msqerror3,error3 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5,5, 1])
#allOut3=net3.sim(DataX)

print "******************* "

net4,inp4,tar4,out4,allOut4,msqerror4,error4 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5,5, 1])
#allOut4=net4.sim(DataX)

print "******************* "

net5,inp5,tar5,out5,allOut5,msqerror5,error5 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5,5, 1])

print "******************* "

net6,inp6,tar6,out6,allOut6,msqerror6,error6 = trainNN(X=DataX.copy(),Y=DataY.copy(),tratio=trainR, inLayer=[[0, 2]], outLayer=[5,5, 1])

W=((1.0/error1) + (1.0/error2) + (1.0/error3) + (1.0/error4) + (1.0/error5) + (1.0/error6) )
w1= (1.0/error1)/W
w2= (1.0/error2)/W
w3= (1.0/error3)/W
w4= (1.0/error4)/W
w5= (1.0/error5)/W
w6= (1.0/error6)/W


Eout=[]
for i in xrange(int ( float(2)/0.002  )):
    x= w1*allOut1[i] + w2*allOut2[i] + w3*allOut3[i] + w4*allOut4[i] + w5*allOut5[i] + w6*allOut6[i]
    if x== 'nan':
        x=DataY[i]
    Eout.append( x )

Eout=np.array(Eout) # in form of [ [],[],[],[], , , , ,]

myarray= np.array( np.array(Eout[:,0])-np.array(DataY[:,0]) )
Eerror=[]

for a in myarray:
    #b=math.pow(a, 2)
    b=abs(a)

    Eerror.append(b)
Eerror=np.array(Eerror)

print Eerror.shape

index=np.arange(0, int (float(2)/0.002), 1)



plt.subplot(2,4,3)
plt.plot(DataX[:,0], allOut1[:,0],)
plt.plot(DataX[index][:,0], msqerror1[index],'r--')
plt.scatter(inp1[:,0],out1[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function1, %Error=(" + str(("{0:.3f}".format(float(msqerror1.sum()/100 )))+ "%)"))

plt.subplot(2,4,4)
plt.plot(DataX[:,0], allOut2[:,0])
plt.plot(DataX[index][:,0], msqerror2[index],'r--')
plt.scatter(inp2[:,0],out2[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function2, %Error=(" + str(("{0:.3f}".format(float(msqerror2.sum()/100 )))+ "%)"))


plt.subplot(2,4,5)
plt.plot(DataX[:,0], allOut3[:,0])
plt.plot(DataX[index][:,0], msqerror3[index],'r--')
plt.scatter(inp3[:,0],out3[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function3, %Error=(" + str(("{0:.3f}".format(float(msqerror3.sum()/100 )))+ "%)"))

plt.subplot(2,4,6)
plt.plot(DataX[:,0], allOut4[:,0])
plt.plot(DataX[index][:,0], msqerror4[index],'r--')
plt.scatter(inp4[:,0],out4[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function4, %Error=(" + str(("{0:.3f}".format(float(msqerror4.sum()/100 )))+ "%)"))

plt.subplot(2,4,7)
plt.plot(DataX[:,0], allOut5[:,0])
plt.plot(DataX[index][:,0], msqerror5[index],'r--')
plt.scatter(inp5[:,0],out5[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function5, %Error=(" + str(("{0:.3f}".format(float(msqerror5.sum()/100 )))+ "%)"))

plt.subplot(2,4,8)
plt.plot(DataX[:,0], allOut6[:,0])
plt.plot(DataX[index][:,0], msqerror6[index],'r--')
plt.scatter(inp6[:,0],out6[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("function6, %Error=(" + str(("{0:.3f}".format(float(msqerror6.sum()/100 )))+ "%)"))



plt.subplot(2,4,1)
plt.plot(DataX[:,0], DataY[:,0])
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("Actual Function")

plt.subplot(2,4,2)
plt.plot(DataX[:,0], Eout[:,0])
plt.plot(DataX[index][:,0], Eerror[index],'r--')
plt.xlim(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1)
plt.ylim(DataY.min()-0.1, DataY.max()+0.1)
plt.xticks((np.arange(DataX[:,0].min()-0.1, DataX[:,0].max()+0.1,0.5)))
plt.yticks((-1,-0.5,0,0.5,1))
plt.title("Ensembled Output, %Error=(" + str(("{0:.3f}".format(float(Eerror.sum())/100 )))+ "%)")


print "runtime = ", (time.time()-ti), " sec"

plt.show()
