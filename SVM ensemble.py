__author__ = 'manuj'

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets

t1=time.time()

def load_data(file_name):

    file = open(file_name)

    in_data_array=[]

    for line in file.readlines():
        new_array=line.split()
        in_data_array.append(new_array)


    in_data_array=np.array(in_data_array)

    in_data_array=in_data_array[:,np.newaxis]

    X1= map(float,in_data_array[:,0,1])
    X2= map(float,in_data_array[:,0,2])

    X=np.c_[X1,X2]
    X=np.array(X)

    Y= map(float,in_data_array[:,0,0])
    Y=np.array(Y)

    return(X,Y)

Xtrain,Ytrain= load_data("DigitsIn")
Xtest,Ytest=   load_data("DigitsOut")


def get_vsall(Y,value):

    #print Y

    #print (Y==value)

    Y[[Y!=value]]=-1.0
    Y[[Y==value]]=1.0

    return Y

Ytrain1=get_vsall(Ytrain.copy(),7)
Ytest1=get_vsall(Ytest.copy(),7)

print (Ytrain1==1).sum()

#print Ytrain

#print (Ytrain==3.00).sum()


h = .02  # step size in the mesh
C = 0.2  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(Xtrain,Ytrain1)
print  "1"
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(Xtrain,Ytrain1)
print  "2"
poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(Xtrain,Ytrain1)
print  "3"
lin_svc = svm.LinearSVC(C=C).fit(Xtrain,Ytrain1)
print  "4"
# create a mesh to plot in
x_min, x_max = Xtrain[:, 0].min() , Xtrain[:, 0].max()
y_min, y_max = Xtrain[:, 1].min() , Xtrain[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
print  "5"

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    print  "6"

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    #plt.scatter(Xtrain[Ytrain1==-1][:, 0], Xtrain[Ytrain1==-1][:, 1], c=Ytrain1[Ytrain1==1], cmap=plt.cm.Paired)
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain1, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

print "Runtime=" +str(time.time()-t1 )


