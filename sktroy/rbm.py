'''
Geoffrey Hinton. "A practical guide to training restricted Boltzmann machines."
Momentum 9.1 (2010): 926.
'''
import numpy as np

class RBM(object):
    '''
    Only binary features are supported
    Minimize:
    P(v, h) = 1/Z*exp(-E(v, h))
    E(v, h) = -a⋅v - b⋅h - v'Wh
    '''
    rho = 0.001
    def __init__(self, cd_num=1):
        self.a = None
        self.b = None
        self.W = None
        self.cd_num = cd_num
    def get_grads(self, Visible, Hidden_score, Visible_recon, Hidden_recon):
        grad_W = Visible.T.dot(Hidden_score) - Visible_recon.T.dot(Hidden_recon)
        grad_a = Visible.sum(0) - Visible_recon.sum(0)
        grad_b = Hidden_score.sum(0) - Hidden_recon.sum(0)
        return grad_a, grad_b, grad_W
    def cd(self, Hidden):
        Visible_recon = self.inv_predict(Hidden)
        Visible_recon = RBM.sample(Visible_recon)
        Hidden_recon_score = self.predict(Visible_recon)
        Hidden_recon = RBM.sample(Hidden_recon_score)
        return Visible_recon, Hidden_recon_score, Hidden_recon

    def fit(self, Visible, hidden_dim):
        num, visible_dim = Visible.shape
        self.a = np.random.randn(visible_dim)
        self.b = np.random.randn(hidden_dim)
        self.W = np.random.randn(visible_dim, hidden_dim)
        for itr in range(500):
            Hidden_score = self.predict(Visible)
            Hidden = RBM.sample(Hidden_score)
            Visible_recon, Hidden_recon_score, Hidden_recon = self.cd(Hidden)
            for n in range(1, self.cd_num):
                Visible_recon, Hidden_recon_score, Hidden_recon = self.cd(Hidden_recon)
            grad_a, grad_b, grad_W = self.get_grads(
                Visible, Hidden_score, Visible_recon, Hidden_recon_score)
            self.a += RBM.rho * grad_a
            self.b += RBM.rho * grad_b
            self.W += RBM.rho * grad_W
        return

    def predict(self, Visible):
        return RBM.sigmoid(Visible.dot(self.W) + self.b)
    def inv_predict(self, Hidden):
        return RBM.sigmoid(Hidden.dot(self.W.T) + self.a)
    @staticmethod
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))
    @staticmethod
    def sample(X):
        Y = np.zeros(X.shape)
        Y[X > np.random.random(X.shape)] = 1
        return Y
