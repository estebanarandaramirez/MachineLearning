import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    powers = np.arange(d+1)[:, np.newaxis]
    Xtilde = x
    for i in range(d):
        Xtilde = np.vstack((Xtilde, x))
    Xtilde = np.power(Xtilde, powers)
    return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    ones = np.ones((1, faces.shape[0]))
    faces = np.reshape(faces, (faces.shape[0], faces.shape[1]**2))
    faces = faces.T
    return np.vstack((faces, ones))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    return np.mean((Xtilde.T.dot(w) - y)**2)/2

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    return ((Xtilde.dot((Xtilde.T.dot(w) - y)))/Xtilde.shape[1]) + ((alpha * w.T.dot(w))/(2 * Xtilde.shape[1]))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = np.random.randn(2305) * 0.01
    for i in range(T):
        w = w - EPSILON*gradfMSE(w, Xtilde, y, alpha)
    return w

def visualize(w, title):
    img = np.reshape(w[:2304], (48,48))
    plt.title(title)
    plt.imshow(img)
    plt.savefig((title + ".png"))

def worstPredictions(w, Xtilde, y):
    differences = list(abs(Xtilde.T.dot(w) - y))
    worst = sorted(differences, reverse=True)[:5]
    worst_info = {}
    for error in worst:
        idx = differences.index(error)
        worst_info[error] = [idx, y[idx], Xtilde[:,idx].T.dot(w)]
    return worst_info

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    mse1tr = fMSE(w1, Xtilde_tr, ytr)
    bias1tr = w1[-1]
    print("MSE W1 Training: ", mse1tr)
    print("Bias W1 Training: ", bias1tr)
    visualize(w1, 'Method1')

    w1test = method1(Xtilde_te, yte)
    mse1te = fMSE(w1, Xtilde_te, yte)
    bias1te = w1test[-1]
    print("\nMSE W1 Test: ", mse1te)
    print("Bias W1 Test: ", bias1te)
    visualize(w1test, 'Method1Test')

    w2 = method2(Xtilde_tr, ytr)
    mse2tr = fMSE(w2, Xtilde_tr, ytr)
    bias2tr = w2[-1]
    print("\nMSE W2 Training: ",mse2tr)
    print("Bias W2 Training: ", bias2tr)
    visualize(w2, 'Method2')

    w2test = method2(Xtilde_te, yte)
    mse2te = fMSE(w2, Xtilde_te, yte)
    bias2te = w2test[-1]
    print("\nMSE W2 Test: ", mse2te)
    print("Bias W2 Test: ", bias2te)
    visualize(w2test, 'Method2Test')

    w3 = method3(Xtilde_tr, ytr)
    mse3tr = fMSE(w3, Xtilde_tr, ytr)
    bias3tr = w3[-1]
    print("\nMSE W3 Training: ",mse3tr)
    print("Bias W3 Training: ",bias3tr)
    visualize(w3, 'Method3')

    w3test = method3(Xtilde_te, yte)
    mse3te = fMSE(w3, Xtilde_te, yte)
    bias3te = w3[-1]
    print("\nMSE W3 Test: ", mse3te)
    print("Bias W3 Test: ", bias3te)
    visualize(w3, 'Method3Test')

    worst = worstPredictions(w3, Xtilde_te, yte)
    print("\nW3 RMSE: ", mse3te**(1/2))
    print("5 most egregious errors: ")
    for i, w in enumerate(worst.items()):
        error, info = w
        visualize(Xtilde_te[:,info[0]], 'Worst' + str(i+1))
        print("  Image {}: Error = {} \t Actual = {} \t Predicted = {}".format(info[0], error, info[1], info[2]))

    x = np.array([2, 3, 4, 5])
    y = np.array([0.59926454, 0.46555766, 0.9368333, 0.6477865])
    poly_w = trainPolynomialRegressor(x, y, 3)
    print("\nPolynomial Regression Weights: ", poly_w)