import numpy as np
import pdb
import unittest
import sys
import os.path as path
from sklearn.metrics import roc_auc_score
sys.path.append(path.join(path.dirname(__file__), "../.."))

from sktroy.optimizers.gd_l1 import fmin_owlqn
from sktroy.optimizers.gd_pgd_l1 import fmin_pgm_l1
from sktroy.optimizers.cgd import fmin_cgd
from sktroy.optimizers.lr_cost_grad import LR
from scipy.optimize import fmin_cg


def get_data():
    w = [
         0.1, 0.2, 0.5,
         *([0]*17)]
    w = [-1, 1]
    w = np.asarray([x*(1 if n%2 else -1) for n, x in enumerate(w)])
    num = 10000
    dim = len(w)

    X = np.random.randn(num, dim)
    y = np.dot(X, w.reshape(dim, 1)) > 0
    return X, y.squeeze(), w


class L1LRTestor(object):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    w = None
    def __init__(self):
        X, y, L1LRTestor.w = get_data()
        num, dim = X.shape
        L1LRTestor.X_train, L1LRTestor.X_test = X[:round(num*0.8), :], X[round(num*0.8):, :]
        L1LRTestor.y_train, L1LRTestor.y_test = y[:round(num * 0.8)], y[round(num * 0.8):]

    def _test_owlqn(self):
        print("LR-owlqn")
        lr = LR(fmin_owlqn)
        lr.fit(L1LRTestor.X_train, L1LRTestor.y_train)
        y_pred = lr.predict(L1LRTestor.X_train)
        print(roc_auc_score(L1LRTestor.y_train, y_pred))
        y_pred = lr.predict(L1LRTestor.X_test)
        print(roc_auc_score(L1LRTestor.y_test, y_pred))

    def _test_pgm(self):
        print("LR-pgm")
        lr = LR(fmin_pgm_l1)
        lr.fit(L1LRTestor.X_train, L1LRTestor.y_train)
        y_pred = lr.predict(L1LRTestor.X_train)
        print(roc_auc_score(L1LRTestor.y_train, y_pred))
        y_pred = lr.predict(L1LRTestor.X_test)
        print(roc_auc_score(L1LRTestor.y_test, y_pred))

    def _test_cgd(self):
        print("cg")
        X = L1LRTestor.X_train
        y = L1LRTestor.y_train

        num, dim = X.shape
        sym_y = np.zeros((num, 1), dtype=int)
        sym_y[y > 0] = 1
        sym_y[y <= 0] = -1
        sym_y = sym_y.reshape(num, 1)

        f = lambda w: LR.cost(X, sym_y, w) + 0.005*np.dot(w, w)
        fprime = lambda w: LR.grad(X, sym_y, w) + 0.01*w
        w = fmin_cg(f, np.zeros(dim), fprime)
        print(w)


if __name__ == "__main__":
    # pdb.set_trace()
    test_l1lr = L1LRTestor()
    test_l1lr._test_pgm()
    test_l1lr._test_owlqn()
    test_l1lr._test_cgd()
