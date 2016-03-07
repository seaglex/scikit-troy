import numpy as np
import unittest
import pdb

from sktroy import rbm

class TestRBM(unittest.TestCase):
    def test_rbm(self):
        # prepare data
        num = 1000
        hidden_dim = 5
        visible_dim = 20
        Hidden = np.zeros((num, hidden_dim))
        Hidden[np.random.randn(num, hidden_dim)<0.6] = 1
        Trans_mat = np.random.randn(hidden_dim, visible_dim)
        Visible_score = Hidden.dot(Trans_mat)
        Visible_score += np.mean(np.abs(Visible_score)) * np.random.randn(num, visible_dim) * 0.05
        Visible = (Visible_score > 0) * 1
        # fit & test
        # pdb.set_trace()
        model = rbm.RBM()
        model.fit(Visible, hidden_dim)
        Hidden_recon = model.predict(Visible)
        Visible_recon = model.inv_predict(Hidden_recon)
        print(np.sum(np.abs(Visible - Visible_recon))/np.sum(np.abs(Visible)))

if __name__ == "__main__":
    unittest.main()