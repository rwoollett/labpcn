
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Demonstration of the Perceptron on the Pima Indian dataset

import matplotlib.pyplot as plt
import numpy as np
import pcn

pima = np.loadtxt('pima-indians-diabetes.data',delimiter=',')

# Plot the first and second values for the two classes
indices0 = np.where(pima[:,8]==0)
indices1 = np.where(pima[:,8]==1)

# plt.ioff()
plt.plot(pima[indices0,4],pima[indices0,5],'go')
plt.plot(pima[indices1,4],pima[indices1,5],'rx')

# Perceptron training on the original dataset
print("Output on original data")
p = pcn.pcn(pima[:,:8],pima[:,8:9])
p.pcntrain(pima[:,:8],pima[:,8:9],0.25,100)
p.confmat(pima[:,:8],pima[:,8:9])
print("------------------")

# Various preprocessing steps
#cap the maxium number of times pregnant to 8
pima[np.where(pima[:,0]>8),0] = 8

#Categorize the age data. <= 30 years is set to 1, 30-40 is set to 2, etc
pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60),7] = 5


#normalize all features
pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)


print ("mean",pima[:,:8]-pima[:,:8].mean(axis=0))
print ("var",pima.var(axis=0))
print ("var",pima[:,:8]/pima[:,:8].var(axis=0))
#print pima.max(axis=0)
#print pima.min(axis=0)

#partition the training and test sets
#select even numbered rows for training and odd rows for test set
trainin = pima[::2,:8] #input features for training
testin = pima[1::2,:8] #input features in the test set
print(testin)
traintgt = pima[::2,8:9] #corresponding class labels for the training set
testtgt = pima[1::2,8:9] #corresponding class labels for the test set

# Perceptron training on the preprocessed dataset
print("Output after preprocessing of data")
p1 = pcn.pcn(trainin,traintgt)
p1.pcntrain(trainin,traintgt,0.25,100)
p1.confmat(testin,testtgt)

print(np.shape(testtgt), np.shape(np.where(testtgt[:,0] == 1)), np.shape(np.where(testtgt[:,0] == 0)))

plt.show()
