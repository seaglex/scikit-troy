import pdb
import unittest
import datetime as dt
import numpy as np
import scipy.sparse as sps

from sktroy import nap

class NuisanceDataGenerator(object):
    EPS = 0.1
    Grp_nums = [2, 2, 2, 3]
    def add_linear_nuisance(self, x, nuisances, num):
        sz = nuisances.shape[0]
        return nuisances[np.random.randint(0, sz, num), :] + x
    def get_group_size(self):
        return NuisanceDataGenerator.Grp_nums[np.random.randint(0, 4)]
    def gen_testcase(self, num, dim, nuisance_dim):
        # the sampels in the same grp is assumed to be similar
        nuisances = 0.5 * np.random.randn(nuisance_dim, dim)
        eps = NuisanceDataGenerator.EPS
        ids = []
        X = []
        id_ = 0
        while True:
            sz = self.get_group_size()
            x = np.random.rand(dim)
            observations = self.add_linear_nuisance(x, nuisances, sz)
            observations += eps*np.random.randn(sz, dim)
            X.append(observations)
            ids.append(list(range(id_, id_+sz)))
            id_ += sz
            if id_ >= num:
                break
        X = np.vstack(X)
        return X, ids

    @staticmethod
    def get_weight_matrix(ids, num):
        data = []
        row_indice = []
        col_indice = []
        for id_ in ids:
            for i in id_:
                for j in id_:
                    if i == j:
                        continue
                    data.append(1)
                    row_indice.append(i)
                    col_indice.append(j)
        data = np.array(data, dtype=np.byte)
        return sps.csr_matrix((data, (row_indice, col_indice)), shape=(num, num))
    @staticmethod
    def _get_dist(inner):
        dist = 0
        num = inner.shape[0]
        if num <= 1:
            return 0
        for n, x in enumerate(inner):
            for m in range(n+1, num):
                diff = inner[m, :] - x
                dist += np.sum(diff**2)
        return dist / (num*(num-1)/2)
    @staticmethod
    def get_inner_dist(X, ids):
        inner_dist = 0
        inner_cnt = 0
        for id_ in ids:
            if len(id_) <= 1:
                continue
            inner_dist += NuisanceDataGenerator._get_dist(X[id_, :])
            inner_cnt += 1
        return inner_dist/inner_cnt
    @staticmethod
    def get_inter_dist(X):
        num, dim = X.shape
        inter_cnt = 0
        inter_dist = 0
        for n in range(max(num, 100)):
            n, m = np.random.randint(0, num, 2)
            inter_cnt += 1
            inter_dist += np.sum((X[n, :] - X[m, :])**2)
        return inter_dist/inter_cnt


class TestNap(unittest.TestCase):
    NAP_DIM = 4

    def setUp(self):
        min_num = 1000
        dim = 100
        genor = NuisanceDataGenerator()
        X, ids = genor.gen_testcase(min_num, dim, TestNap.NAP_DIM)
        W = NuisanceDataGenerator.get_weight_matrix(ids, X.shape[0])
        self.X, self.ids, self.W = X, ids, W
        self.inner_dist = NuisanceDataGenerator.get_inner_dist(X, ids)
        self.inter_dist = NuisanceDataGenerator.get_inter_dist(X)
    def test_sparse_weights_speed(self):
        beg = dt.datetime.now()
        naper = nap.NAP()
        naper.fit(self.X, self.W, TestNap.NAP_DIM)
        newX = naper.predict(self.X)
        end = dt.datetime.now()
        dst_inner_dist = NuisanceDataGenerator.get_inner_dist(newX, self.ids)
        dst_inter_dist = NuisanceDataGenerator.get_inter_dist(newX)
        print("Sparse weights computation: {0}s".format((end-beg).total_seconds()))
        print("\tInner\tInter")
        print("Before_NAP\t{0}\t{1}".format(self.inner_dist, self.inter_dist))
        print("After_NAP\t{0}\t{1}".format(dst_inner_dist, dst_inter_dist))
        self.assertLess(dst_inner_dist/dst_inter_dist, 0.2 * self.inner_dist/self.inter_dist)
    def test_dense_weights_speed(self):
        W = self.W.todense()
        # pdb.set_trace()
        beg = dt.datetime.now()
        naper = nap.NAP()
        naper.fit(self.X, W, TestNap.NAP_DIM)
        newX = naper.predict(self.X)
        end = dt.datetime.now()
        dst_inner_dist = NuisanceDataGenerator.get_inner_dist(newX, self.ids)
        dst_inter_dist = NuisanceDataGenerator.get_inter_dist(newX)
        print("Dense weights computation: {0}s".format((end-beg).total_seconds()))
        print("\tInner\tInter")
        print("Before_NAP\t{0}\t{1}".format(self.inner_dist, self.inter_dist))
        print("After_NAP\t{0}\t{1}".format(dst_inner_dist, dst_inter_dist))
        self.assertLess(dst_inner_dist/dst_inter_dist, 0.2 * self.inner_dist/self.inter_dist)

if __name__ == "__main__":
    unittest.main()
