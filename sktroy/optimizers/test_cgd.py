'''
Even the naive implementation of cgd is better than steepest gradient descend in bad situation
'''
import numpy as np
from scipy.optimize import fmin_cg
import pdb
import sys
import os.path as path
from sklearn.metrics import roc_auc_score
sys.path.append(path.join(path.dirname(__file__), "../.."))

from sktroy.optimizers.cgd import fmin_cgd
from sktroy.optimizers.gd_l1 import fmin_owlqn

def f(xs):
    # print("f called")
    return 0.5*((xs[0]-1)**2) + 0.5*10000*((xs[1]-1)**2)

def g(xs):
    # print("g called")
    return np.asarray( [xs[0]-1, 10000*(xs[1]-1)] )

x0 = np.asarray([0, 0])

class CgdTestor(object):
    def _test_gd(self):
        print("gd")
        x, y, x_hist, y_hist = fmin_owlqn(f, x0, g, 0, 0.0001, 100)
        print(y_hist)

    def _test_cgd(self):
        x, y, x_hist, y_hist = fmin_cgd(f, x0, g, 100)
        print("cgd")
        print(y_hist)

if __name__ == "__main__":
    # pdb.set_trace()
    testor = CgdTestor()
    testor._test_cgd()
    testor._test_gd()
