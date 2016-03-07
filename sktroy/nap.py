'''
Alex Solomonoff, William M. Campbell, and Carl Quillen.
"Nuisance attribute projection." Speech Communication (2007): 1-73.
'''
import numpy as np

class NAP:
    def __init__(self):
        self.V = None
    def fit(self, X, W, k):
        num, dim = X.shape
        k = min(k, dim)
        w1 = W.sum(1)
        # density matrix complexity: dim * num * (num + dim)
        # sparse matrix complexity: dim * (dim * num + num_nonzero)
        M = np.dot(X.T, W.dot(X))
        M -= np.dot(np.multiply(X.T, w1.reshape(1, num)), X)
        eigen_values, eigen_vectors = np.linalg.eigh(M)
        abs_eigen_values = np.abs(eigen_values)
        index = np.argsort(abs_eigen_values)
        self.V = np.asarray(eigen_vectors[:, index[-k:]])
    def predict(self, X):
        '''
        :return: (E - V⋅V')⋅X' => X⋅(E - V⋅V')
        '''
        # since k << dim, it's more efficient to calculate step by step
        return X - np.dot(np.dot(X, self.V), self.V.T)


