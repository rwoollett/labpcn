'''
Created on Oct 7, 2012

@author: srmarsla
'''

import matplotlib.pyplot as plt
import numpy as np
import pcn
import pickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()

#lets plot the first digit in the training set
plt.imshow(np.reshape(train_set[0][200,:],[28,28]))
print("The correct digit for the first image in the train set is :", train_set[1][0])

nread = 200
# Just use the first few images
train_in = train_set[0][:nread,:]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((nread,10))
for i in range(nread):
    train_tgt[i,train_set[1][i]] = 1

test_in = test_set[0][:nread,:]
test_tgt = np.zeros((nread,10))
for i in range(nread):
    test_tgt[i,test_set[1][i]] = 1

# Train a Perceptron on training set
p = pcn.pcn(train_in, train_tgt)

inputs_bias = np.concatenate((train_in,-np.ones((np.shape(train_in)[0],1))), axis=1)

p.pcntrain(train_in, train_tgt,0.25,100)

# This isn't really good practice since it's on the training data, 
# but it does show that it is learning.
p.confmat(train_in,train_tgt)

# Now test it
p.confmat(test_in,test_tgt)

plt.show()