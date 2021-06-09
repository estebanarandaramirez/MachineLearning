import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return A.dot(B) - C

def problem3 (A, B, C):
    return A*B + C.T

def problem4 (x, S, y):
    return x.T.dot(S.dot(y))

def problem5 (A):
    return np.zeros((A.shape))

def problem6 (A):
    return np.ones((A.shape[0],1))

def problem7 (A, alpha):
    return A + alpha*np.eye(A.shape[0])

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    A = A[np.nonzero(A)]
    A = A[np.where((A >= c) & (A <=d))]
    return np.mean(A)

def problem11 (A, k):
	_, eigvals = np.linalg.eig(A)
	return eigvals[:, A.shape[0] - k:]

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return (np.linalg.solve(A.T, x.T)).T

A = np.array([[1,0,0],[0,-2,0],[0,0,5]])
x = np.array([[2], [2], [2]])
y = np.array([[1], [1], [1]])
print(problem4(x, A, y))
print(problem5(A))
print(problem6(A))
print(problem7(A, 1))
print(problem8(A, 1, 1))
print(problem9(A, 2))
print(problem10(A, 1, 5))
print(problem11(A, 2))
print(problem12(A, x))
x = x.T
print(problem13(A, x))