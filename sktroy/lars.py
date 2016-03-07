'''
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani.
"Least angle regression." The Annals of statistics 32.2 (2004): 407-499.
'''
import numpy as np

from sktroy import mat_utils

class LARS(object):
    def __init__(self):
        self.weights_histories = []
    def fit(self, X, y):
        num, dim = X.shape
        weights = np.zeros(dim)
        # initialization
        cors = np.dot(X.T, y)
        signs = np.ones(dim)
        signs[cors < 0] = -1
        cors[cors < 0] *= -1
        I = np.argmax(cors)
        actives = []
        # step-wise select new direction
        iCXX = np.zeros(())
        for i in range(dim):
            if i == 0:
                iCXX = 1.0 / np.dot(X[:, I].reshape(1, num), X[:, I].reshape(num, 1))
                actives.append(I)
            else:
                # incrementally get the inverse correlation matrix of selcted X
                z = X[:, I] * signs[I]
                czz = np.dot(z, z)
                czX = np.dot(z.reshape(1, num), X[:, actives]*signs[actives])
                iCXX = mat_utils.incremental_inv(iCXX, czX.T, czX, czz)
                actives.append(I)
            max_cor = cors[I]
            a = 1/np.sqrt(np.sum(iCXX))
            omega = a*np.sum(iCXX, 1)
            direction = np.dot(X[:, actives]*signs[actives], omega)
            # tooptimize: cor_desc[actives) == a, no need to compute
            cor_descs = np.dot((X*signs).T, direction)
            # select the next dim to active set
            selection = (I, cors[I]/cor_descs[I])
            for n in range(dim):
                # tooptimize: inefficient search operation
                if n in actives:
                    continue
                gamma1 = (max_cor-cors[n]) / (a-cor_descs[n])
                gamma2 = (max_cor+cors[n]) / (a+cor_descs[n])
                if gamma1 >= 0 and gamma1 < selection[1]:
                    selection = (n, gamma1)
                if gamma2 >= 0 and gamma2 < selection[1]:
                    selection = (n, gamma2)
            # update x'(y-mu)
            I = selection[0]
            cors -= cor_descs*selection[1]
            # tooptimize: only update the inactive
            if i < dim - 1:
                # in the last iteration, cors ~== 0, avoid numeric problem
                signs[cors < 0] *= -1
                cors[cors < 0] *= -1
            weights = weights.copy()
            weights[actives] += selection[1] * omega * signs[actives]
            self.weights_histories.append(weights)
        assert np.allclose(cors, 0)