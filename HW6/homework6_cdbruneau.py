# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_DNp6GFG-41nzKJnsTei81hiQ0y5EiQ3
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy
import random
import math

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w): 
  W1 = w[:NUM_INPUT * NUM_HIDDEN].reshape((NUM_INPUT, NUM_HIDDEN))
  #print(W1.shape)
  #(784,40)

  b1 = w[NUM_INPUT * NUM_HIDDEN: NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN].reshape((NUM_HIDDEN))
  #print(b1.shape)
  #(40)

  W2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN:NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT].reshape((NUM_HIDDEN, NUM_OUTPUT))
  #print(W2.shape)
  #(40,10)

  b2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT:].reshape(NUM_OUTPUT)
  #print(b2.shape)
  #(10)
   
  return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.4
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
  #flattern each into 1d
  W1 = W1.flatten()
  b1 = b1.flatten()
  W2 = W2.flatten()
  b2 = b2.flatten()
  #concatenate and return vector of all values
  w = np.concatenate((W1, b1, W2, b2))
  #print("packed size")
  #print(w.shape)
  return w
 

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    images = images/255.0
    return images, labels

#Need to have function for propagation given that it will be called multiple times
def forwardProp(X, w):
  #unpack our w to get weights/biases
  W1, b1, W2, b2 = unpack(w)

  #According to hw 6
  #z(1)=W(1)x+b(1)
  z1 = np.dot(X,W1) + b1
  #h(1)= relu(z(1)))
  h1 = relu(z1)
  #z(2)=W(2)h(1)+b(2)
  z2 = np.dot(h1, W2)+b2
  #??y=g(x) = softmax(z(2))
  yHat = softmax(z2)

  return z1, h1, z2, yHat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):    
  W1, b1, W2, b2 = unpack(w)

  #forward propogation
  z1, h1, z2, yHat = forwardProp(X, w)

  #calculate the CE loss
  #Function:
  #fCE(W(1),b(1),W(2),b(2)) =???1nn???i=110???k=1y(i)klog??y(i)k

  yTrans = yHat.T

  sum = np.sum(Y * np.log(yTrans), axis = 1)

  cost  = np.mean(sum)

  #cost needs to detract, so get negative value
  
  cost = cost*-1


  return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):    
  W1, b1, W2, b2 = unpack(w) 

  #forward propogation
  z1, h1, z2, yHat = forwardProp(X, w)

  yTrans = yHat.T

  yDiff = yTrans-Y
 # print("yDiff")
  #print(yDiff.shape)

  g = ((yDiff).dot(W2.T)) * reluDer(z1) 

  #update values
  #???W(2)fCE=(??y???y)h(1)>
  W2Der = np.atleast_2d(yDiff).T.dot(h1).T

  #???b(2)fCE=    (??y???y)
  b2Der = np.mean(yDiff, axis = 0)
 # b2Der = yDiff

  #???W(1)fCE=gx>
  W1Der = X.T.dot(np.atleast_2d(g))

  #???b(1)fCE=g
  b1Der = np.mean(g, axis = 0)
  #b1Der = g

  #need to pack uor new variables

  gCE =  pack(W1Der, b1Der, W2Der, b2Der)

  return gCE


def softmax(x):

    #get exponent
    exp = np.exp(x-np.max(x))

    #sum the exponents
    sum = (exp).sum(axis = 1)

    #tranpose exp
    trans = exp.T

    #return normalized vector
    norm = trans / sum

    return norm

def oneHot(labels):
    return np.eye(labels.max()+1)[labels]


def transformOH(yhat):
    yhat = yhat.T

    yHatTrans= np.zeros_like(yhat)

    yHatTrans[np.arange(len(yhat)), yhat.argmax(1)] = 1

    return yHatTrans

def relu(x):
    x = x.copy()
    x[x<=0] = 0
    return x

def reluDer(x):
  x = x.copy()
  x[x <= 0] = 0
  x[x > 0] = 1
  return x


def findBestHyperparameters(trainX, trainY, testX, testY, w):

    #array of hyperparameters, suggested amounts in hw 6
    epochVals = [10, 20, 30, 40, 50, 100]

    batchVals = [16, 32, 64, 128, 256]

    learningVals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    hiddenLayers = [30, 40, 50]

    #regularization strength?

    #set defualt values, loss and accuracy to unachievable worst/best cases
    currVals = {'Epoch': 0, 'LearningRate': 0.0, 'Batch': 0, 'Loss': 100.0, 'Accuracy': 0.0}

    
    #Set aside 20% of data to train hyperparameter
    permute = np.random.permutation(math.floor(trainX.shape[0]*0.2))
    xPermute = trainX[permute]
    yPermute = trainY[permute]

    print(xPermute.shape)
    print(yPermute.shape)

    #choose 10 random combinations of hyper parameters, and run the training function on them to determine loss and accuracy.
    for run in range(10):

        epochs = random.choice(epochVals)

        batchSize = random.choice(batchVals)

        learningRate = random.choice(learningVals)

        #report current settings
        print("Random Hyper-Parameters Set ", run+1, "Epochs:", epochs, " Learning Rate:", learningRate, " Batch Size:", batchSize)

        #train the model with the set parameters
        loss, acc = train(trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, False)


        #New best combination?
        if (acc > currVals.get('Accuracy')) and (loss < currVals.get('Loss')):
            currVals.update(Epoch = epochs, Batch = batchSize, LearningRate = learningRate, Loss = loss, Accuracy = acc)

    
    #Return best result
    print("Best Result")
    print(currVals)

    return currVals.get('Epoch'), currVals.get('Batch'), currVals.get('LearningRate')







# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, finalTrain = True):

    #unpack our weights and biases
    W1,b1,W2,b2 = unpack(w)


    for z in range(epochs):
        
        #randomize our data
        permute = np.random.permutation(trainX.shape[0])
        xPermute = trainX[permute]
        yPermute = trainY[permute]


        X = xPermute
        y = yPermute


        #iterate through the data by going up by batch size to obtain gradient, weight, etc
        for n in range(0, X.shape[0], batchSize):

            #our group of data and labels will continue to increase as we add more and more batches
            currXs = X[n:n + batchSize]
            currYs = y[n:n + batchSize]

            #what is our gradient term?
            gCE = gradCE(currXs, currYs, w) * (1-0.1) + 0.1*w 

            #update values, subtract gradient multiplied with learning rate
            #our weight is now proportionally changed
            w -= gCE*learningRate


        #determine the loss
        loss = fCE(X,y,w)
        wCopy = w.copy()


        #only print out last 20 epocs if its the actual training as opposed to hyper paramters
        if finalTrain and (epochs-z <= 20):
            print("Epoch #: ", z+1, "CE Loss: ", np.mean(loss))

    #Need to get yHat to measure our accuracy
    z1 , h1, z1, yHat =  forwardProp(testX, w)

    yHat = transformOH(yHat)

    #calculate acc and loss
    acc= np.mean(testY==yHat)

    loss = fCE(trainX, trainY, w)

    #report results and return them
    print("CE Loss:", loss)
    print("Percent Accuracy:", acc, "\n")

    return loss, acc







if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    

    trainY = oneHot(trainY)

    testY = oneHot(testY)


    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                     lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                     w))


    #Check difference with individual vector components, b2 likely being the smallest difference
    approxF = scipy.optimize.approx_fprime(w, lambda w_: fCE(trainX[idxs,:], trainY[idxs,:], w_), 1e-8)
    w1f, b1f, w2f, b2f = unpack(approxF)
    
    gResult = gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w)
    w1g, b1g, w2g, b2g = unpack(gResult)

    b2Diff = b2f-b2g 
    print("b2 diff:")
    print(b2Diff, "\n")

    # Train the network using SGD.
  
    epoch, batchNum, learnRate = findBestHyperparameters(trainX, trainY, testX, testY, w)

    #epoch = int(epoch)
    #batchNum = int(batchNum)
    #learnRate = float(learnRate)
    
    loss, acc = train(trainX, trainY, testX, testY, w, epoch, batchNum, learnRate, True)

    #Return final training

    print("\n")
    print("Final Test Results")
    print("CE Loss:", loss)
    print("Percent Accuracy:", acc, "\n")