import numpy as np
import unittest

from sktroy import lars

class TestLars(unittest.TestCase):
    @staticmethod
    def _inner_test(self, X, y):
        lars = lars.LARS()
        lars.fit(X, y)
        last_weights = lars.weights_histories[-1]
        ols_weights = np.dot(
            np.linalg.pinv(np.dot(X.T, X)),
            np.dot(X.T, y)
        )
        diff = lars.weights_histories[-1] - ols_weights
        self.assertAlmostEqual(np.sum(diff), 0)

    def test_sign_change(self):
        X = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, -1, 0]
        ], dtype=np.float64).T
        X[:, 2] *= 1.0 / np.sqrt(3)
        y = np.array([1, 0.6, 0.2, 0.1])
        return X, y

    def test_random(self):
        N = 500
        D = 100
        X = np.random.randn(N, D)
        y = np.random.randn(N)
        return X, y

if __name__ == "__main__":
    unittest.main()
