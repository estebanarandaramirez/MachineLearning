import pandas
import numpy as np
# from homework3_earandaramirez import *

def softmaxRegression(trainingImages, trainingLabels, epsilon = None, batchSize = None):
    w = np.random.randn(4, 2) * .1
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
    return w

def fCE(yhat, y):
    crossEntropy =  -1 / y.shape[0] * np.sum(y * np.log(yhat), axis = 1)
    return crossEntropy

def gradfCE(X, yhat, y):
    grad = -1 / np.shape(X)[0] * np.dot(X.transpose(), y - yhat)
    return grad

def classify(X, w):
    z = softMax(X, w)
    return z.argmax(axis = 1)

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

def oneHotEncode(labels):
    return np.eye(labels.max()+1)[labels]

def designMatrix(sex, Pclass, SibSp):
    return np.vstack((sex, Pclass, SibSp, np.ones_like(sex)))

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    training_y = d.Survived.to_numpy()
    training_y = oneHotEncode(training_y)
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy()
    training_X = designMatrix(sex, Pclass, SibSp).T

    # Train model using part of homework 3.
    W = softmaxRegression(training_X, training_y, epsilon=0.1, batchSize=100)

    # Load testing data
    d = pandas.read_csv("test.csv")
    pID = d.PassengerId.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy()
    testing_X = designMatrix(sex, Pclass, SibSp).T

    # Compute predictions on test set
    predictions = classify(testing_X, W)

    # Write CSV file of the format:
    # PassengerId, Survived
    export_array = np.vstack((pID, predictions)).T
    np.savetxt('predictions.csv', export_array, delimiter=',', header='PassengerId,Survived', comments='', fmt='%d')