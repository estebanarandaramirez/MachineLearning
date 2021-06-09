import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    w = np.random.randn(trainingImages.shape[1], trainingLabels.shape[1]) * .1
    X, y = randomizeData(trainingImages, trainingLabels)
    epochs = 300
    printFlag = True
    for i in range(epochs):
        for j in range(0, X.shape[0], batchSize):
            X_batch = X[j:j + batchSize]
            y_batch = y[j:j + batchSize]

            z = softMax(X_batch, w)
            loss = fCE(z, y_batch)
            w = w - (epsilon * gradfCE(X_batch, z, y_batch))

            if i == (epochs-1) and j >= (X.shape[0]-batchSize*20):
                if printFlag:
                    print("Last 20 mini-batches")
                    printFlag = False
                print("Loss: {}".format(np.mean(loss)))
    return w

def fCE(yhat, y):
    crossEntropy =  -1 / y.shape[0] * np.sum(y * np.log(yhat), axis = 1)
    return crossEntropy

def gradfCE(X, yhat, y):
    grad = -1 / np.shape(X)[0] * np.dot(X.transpose(), y - yhat)
    return grad

def classify(X, w):
    z = softMax(X, w)
    encoded_idx = z.argmax(axis = 1)
    return np.eye(encoded_idx.max()+1)[encoded_idx]

def percentAccuracy(yhat, y):
    accuracy = 0.0
    for i in range(y.shape[0]):
        if (y[i] == yhat[i]).all():
            accuracy += 1
    return accuracy/y.shape[0]

def softMax(X, w):
    z = np.dot(X, w)
    z_exp = np.exp(z)
    sum_exp = z_exp.sum(axis = 1)
    normalized = (z_exp.transpose() / sum_exp).transpose()
    return normalized

def randomizeData(X, y):
    indexes = np.random.permutation(X.shape[0])
    X = X[indexes]
    y = y[indexes]
    return X, y

def normalizeandAppend1s(faces):
    faces = faces/255
    ones = np.ones((faces.shape[0], 1))
    return np.hstack((faces, ones))

def oneHotEncode(labels):
    return np.eye(labels.max()+1)[labels]

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = normalizeandAppend1s(trainingImages)
    trainingLabels = oneHotEncode(trainingLabels)
    testingImages = normalizeandAppend1s(testingImages)
    testingLabels = oneHotEncode(testingLabels)

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)
    
    yhat = classify(testingImages, W)
    accuracy = percentAccuracy(yhat, testingLabels)
    print("Testing CE Loss: ", np.mean(fCE(softMax(testingImages, W), testingLabels)))
    print("Testing PC Accuracy: ", accuracy)
    
    # Visualize the vectors
    for i in range(W.shape[1]):
        plotW = np.reshape(W[:784,i], (28,28))
        plt.title("Weight"+str(i+1))
        plt.imshow(plotW)
        plt.savefig("Weight{}.png".format(i+1))