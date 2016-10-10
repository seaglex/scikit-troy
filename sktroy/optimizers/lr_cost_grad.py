'''
Proximal gradient method
'''
import numpy as np


class LR(object):
    _min_pr = np.exp(-10)

    @staticmethod
    def cost(X, sym_y, w0):
        # print("lr-cost")
        N, dim = X.shape
        sym_y = sym_y.reshape(N, 1)
        fx = 1.0 / (1.0 + np.exp(-np.dot(X, w0.reshape(dim, 1)) * sym_y))
        ln_fx = fx.copy()
        ln_fx[fx <= LR._min_pr] = LR._min_pr
        ln_fx = np.log(ln_fx)
        return -(1 / N) * ln_fx.sum()

    @staticmethod
    def grad(X, sym_y, w0):
        # print("lr-grad")
        N, dim = X.shape
        sym_y = sym_y.reshape(N, 1)
        fx = 1.0 / (1.0 + np.exp(-np.dot(X, w0.reshape(dim, 1)) * sym_y))
        return (1.0 / N) * (np.dot((-(1 - fx) * sym_y).reshape(1, N), X)).squeeze()

    @staticmethod
    def _predict(X, w):
        dim = len(w)
        fx = 1.0 / (1.0 + np.exp(-np.dot(X, w.reshape(dim, 1))))
        return fx

    def __init__(self, optimize):
        self._w = None
        self._optimize = optimize

    def fit(self, X, y, *, T=1e-2, eps=1e-1, max_iteration=100):
        num, dim = X.shape
        sym_y = np.zeros((num, 1), dtype=int)
        sym_y[y > 0] = 1
        sym_y[y <= 0] = -1
        sym_y = sym_y.reshape(num, 1)
        f = lambda w: LR.cost(X, sym_y, w)
        fprime = lambda w: LR.grad(X, sym_y, w)
        w, cost, w_hist, cost_hist = self._optimize(f, np.zeros(dim), fprime, T, eps, max_iteration)
        self._w = w
        for n, w_ in enumerate(w_hist):
            print(n, cost_hist[n], sum(1 for x in w_hist[n] if x != 0))
        print(w)

    def predict(self, X):
        return LR._predict(X, self._w)
