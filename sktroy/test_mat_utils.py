import unittest
import numpy as np

from sktroy import mat_utils

class TestMatUnits(unittest.TestCase):
    def test_incremental_inv(self):
        N = 4
        A = np.random.randn(N, N)
        invSubA = np.linalg.inv(A[:N-1, :N-1])
        inv = mat_utils.incremental_inv(invSubA, A[:-1, -1], A[-1, :-1], A[-1, -1])
        zero = A.dot(inv) - np.eye(N)
        self.assertAlmostEqual(np.sum(zero), 0)

if __name__ == "__main__":
    unittest.main()