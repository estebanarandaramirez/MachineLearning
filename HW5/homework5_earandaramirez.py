import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (x):
    r = x[:,0]
    a = x[:,1]
    poly = np.array([np.sqrt(3)*a, np.sqrt(3)*a**2, a**3, np.sqrt(3)*r, np.sqrt(6)*r*a,
     np.sqrt(3)*r*a**2, np.sqrt(3)*r**2, np.sqrt(3)*a*r**2, r**3]).T
    return np.hstack((poly, np.ones((100, 1))))

def kerPoly3 (x, xprime):
    K = (1 + np.dot(x, xprime))**3
    return K

def showPredictions (title, svm, X, xx, yy, transformed=None):  # feel free to add other parameters if desired
    plt.clf()

    if not isinstance(transformed, np.ndarray):
        boundary = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        boundary = boundary.reshape(xx.shape)
        plt.contourf(xx, yy, boundary, colors=("#1f77b4", "#ff9900"), alpha=0.5)
        Z = svm.predict(X) 
    else:
        Z = svm.predict(transformed)

    idxsNeg = np.nonzero(Z == -1)[0]
    idxsPos = np.nonzero(Z == 1)[0]
    nolung = plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1], color="#1f77b4")
    lung = plt.scatter(X[idxsPos, 0], X[idxsPos, 1], color="#ff9900")

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([nolung, lung], [ "No lung disease", "Lung disease" ], loc="upper right")
    plt.title(title)
    plt.savefig("{} Scatter.png".format(title))

if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.title("Scatter-plot")
    plt.savefig("Scatter.png")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X, xx, yy)

    # (b) Poly-3 using explicit transformation phiPoly3
    svmPhiPoly = sklearn.svm.SVC(kernel='linear', C=0.01)
    poly = phiPoly3(X)
    svmPhiPoly.fit(poly, y)
    showPredictions("PhiPoly3", svmPhiPoly, X, xx, yy, poly)
    
    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    svmKerPoly = sklearn.svm.SVC(kernel='precomputed', C=0.01)
    Xprime = X.T
    K = kerPoly3(X, Xprime)
    svmKerPoly.fit(K, y)
    showPredictions("KerPoly3", svmKerPoly, X, xx, yy, K)

    # (d) Poly-3 using sklearn's built-in polynomial kernel
    svmPoly = sklearn.svm.SVC(kernel='poly', gamma=1, coef0=1, degree=3, C=0.01)
    svmPoly.fit(X, y)
    showPredictions("Poly sklearn", svmPoly, X, xx, yy)

    # (e) RBF using sklearn's built-in polynomial kernel
    svmRBF1 = sklearn.svm.SVC(kernel='rbf', gamma=0.1, C=1.0)
    svmRBF1.fit(X, y)
    showPredictions("RBF 0.1", svmRBF1, X, xx, yy)
    svmRBF2 = sklearn.svm.SVC(kernel='rbf', gamma=0.03, C=1.0)
    svmRBF2.fit(X, y)
    showPredictions("RBF 0.03", svmRBF2, X, xx, yy)