import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy
import random

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w[:NUM_INPUT * NUM_HIDDEN].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = w[NUM_INPUT * NUM_HIDDEN: NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN].reshape((NUM_HIDDEN))
    W2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN:NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT].reshape((NUM_HIDDEN, NUM_OUTPUT))
    b2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT:].reshape(NUM_OUTPUT)
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    W1vect = W1.flatten()
    b1vect = b1.flatten()
    W2vect = W2.flatten()
    b2vect = b2.flatten()
    w = np.concatenate((W1vect, b1vect, W2vect, b2vect))
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    return images, labels

#Calculation Process, will be helper later for training
def fPropagation(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = X.dot(W1) + b1
    h1 = ReLU(z1)
    z2 = h1.dot(W2) + b2
    yhat = softMax(z2)

    return z1,h1,z2,yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = fPropagation(X, w)
    cost = np.mean(-1 * np.sum(Y * np.log(yhat.transpose()), axis = 1))
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = fPropagation(X, w)

    db2 = np.mean(yhat.transpose() - Y, axis = 0)
    dW2 = np.atleast_2d(yhat.transpose() - Y).transpose().dot(h1).transpose()
    g = ((yhat.transpose() - Y).dot(W2.transpose())) * derReLU(z1)
    db1 = np.mean(g, axis = 0)
    dW1 = X.transpose().dot(np.atleast_2d(g))

    grad = pack(dW1, db1, dW2, db2)

    return grad

def permutationData(X, y):
    permute = np.random.permutation(X.shape[0])
    Xpermute = X[permute]
    ypermute = y[permute]
    return Xpermute, ypermute

def softMax(z):
    Zexp = np.exp(z)
    sumZexp = np.atleast_2d(Zexp).sum(axis = 1)
    normalized = (Zexp.transpose() / sumZexp)
    return normalized

def ReLU(z):
    z[z<=0] = 0
    return z

def derReLU(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def oneHotEncode(labels):
    return np.eye(labels.max()+1)[labels]

def transform(yhat):
    yhat = yhat.transpose()
    onehot = np.zeros_like(yhat)
    onehot[np.arange(len(yhat)), yhat.argmax(1)] = 1
    return onehot

def percentAccuracy(y, yhat):
    return np.mean(y==yhat)

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, printvar = True):
    b = 0.1
    W1,b1,W2,b2 = unpack(w)

    for i in range(epochs):
        X, y = permutationData(trainX, trainY)
        for j in range(0, X.shape[0], batchSize):
            X_batch = X[j:j + batchSize]
            y_batch = y[j:j + batchSize]
            grad = w * b + gradCE(X_batch, y_batch, w) * (1 - b)
            w = w - (learningRate * grad)
        cost = fCE(X,y,w)

        if printvar and (epochs - i) <= 20:
            print("Epoch:", i + 1, " Loss:", np.mean(cost))

    __, __, __, yhat = fPropagation(testX, w)
    yhat = transform(yhat)
    accuracy = percentAccuracy(testY, yhat)
    loss = fCE(trainX, trainY, w)
    print("CE Loss:", loss)
    print("Accuracy:", accuracy, "\n")

    return loss, accuracy

def findBestHyperparameters(trainX, trainY, testX, testY, w):
    epochTrain = [10, 20, 30, 40, 50, 75, 100]
    batchSizeTrain = [16, 32, 64, 128, 256]
    learningRateTrain = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    optimal = {'Epoch': 0, 'Batch': 0, 'learnRate': 0.0, 'Loss': 10.0, 'Accuracy': 0.0}
    original_w = copy.deepcopy(w)

    for i in range(10):
        w = original_w
        epochs = random.choice(epochTrain)
        learningRate = random.choice(learningRateTrain)
        batchSize = random.choice(batchSizeTrain)
        print("Hyper parameters for run ", (i+1), "- Epochs:", epochs, " Learning Rate:", learningRate, " Batch Size:", batchSize)
        loss, accuracy = train(trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, printvar = False)

        if loss < optimal.get('Loss') and accuracy > optimal.get('Accuracy'):
            optimal.update(Epoch = epochs, Batch = batchSize, learnRate = learningRate, Loss = loss, Accuracy = accuracy)

    print("Best hyper parameters on test data:")
    print(optimal)
    return optimal.get('Epoch'), optimal.get('Batch'), optimal.get('learnRate'), original_w

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

    # Check that the gradient is correct on just a few examples (randomly drawn).
    trainX = trainX/255
    trainY = oneHotEncode(trainY)
    testX = testX/255
    testY = oneHotEncode(testY)

    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))

    approxF = scipy.optimize.approx_fprime(w, lambda w_: fCE(trainX[idxs,:], trainY[idxs,:], w_), 1e-8)
    w1f, b1f, w2f, b2f = unpack(approxF)
    
    gResult = gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w)
    w1g, b1g, w2g, b2g = unpack(gResult)

    b2Diff = b2f-b2g 
    print("b2 diff:")
    print(b2Diff, "\n")

    # Train the network using SGD.
    cutoff = int(trainX.shape[0]*0.2)
    validationX = trainX[:cutoff]
    validationY = trainY[:cutoff]
    trainX_subset = trainX[cutoff:]
    trainY_subset = trainY[cutoff:]

    epochs, batchSize, learningRate, w = findBestHyperparameters(trainX_subset, trainY_subset, validationX, validationY, w)
    _ = train(trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, printvar = True)
