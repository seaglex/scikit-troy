import numpy as np


class MAPELinearRegression(object):
    EPSILON = 1e-6
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
        return

    def predict(self, X):
        results = np.dot(X, self.coef_) + self.intercept_
        return results.squeeze()

    def fit(self, X, y, coef=None, intercept=None):
        num, dim = X.shape
        y = y.reshape(num, 1)
        if not coef:
            w = np.zeros((dim, 1))
        else:
            w = coef.reshape(dim, 1)
        if not intercept:
            b = 0
        else:
            b = intercept
        for itr in range(100000):
            if self._fit_intercept:
                predictions = np.dot(X, w) + b
                errors = 1 - predictions / y
                loss = np.sum(np.abs(errors)) / num
                semi_grads = - 1 / y
                new_b = self.get_optimal(b, errors, semi_grads)
                print("itr {0}, b: loss {1}({2}=>{3})".format(itr, loss, b, new_b))
                b = new_b
            for n in range(dim):
                predictions = np.dot(X, w) + b
                errors = 1 - predictions / y
                loss = np.sum(np.abs(errors)) / num
                semi_grads = -X[:, [n]] / y
                new_w = self.get_optimal(w[n], errors, semi_grads)
                print("itr {0}, w[{1}]: loss {2} ({3}=>{4})".format(itr, n, loss, w[n], new_w))
                w[n] = new_w
        predictions = np.dot(X, w) + b
        errors = 1 - predictions / y
        loss = np.sum(np.abs(errors)) / num
        print("final loss:", loss)
        self.coef_, self.intercept_ = w, b

    def get_optimal(self, x0, errors, semi_grads):
        # ignore un-achievable points & record the gradients
        achievable_indexes = semi_grads != 0
        grad = np.dot(np.sign(errors[~achievable_indexes]), semi_grads[~achievable_indexes])
        dists = -errors[achievable_indexes] / semi_grads[achievable_indexes]
        indexes = dists.argsort()
        dists = dists[indexes]
        semi_grads = semi_grads[achievable_indexes][indexes]
        if not len(semi_grads):
            return x0
        # minimize x, all positive errors ( assume smaller x, larger error)
        grad += np.sum(semi_grads)
        if grad >= 0.0:
            return x0 + dists[0] - MAPELinearRegression.EPSILON
        last_semi_grad = 0.0
        for n, semi_grad in enumerate(semi_grads):
            # 假设x当前在x0 + dists[n-1]位置
            if n > 0:
                # if grad - last_semi_grad > 0.0:
                #     return x0 + dists[n-1]
                if grad - last_semi_grad >= 0.0:
                    return x0 + dists[n-1] + MAPELinearRegression.EPSILON
            new_grad = grad - semi_grad - last_semi_grad
            if new_grad > 0.0:
                return x0 + dists[n] - MAPELinearRegression.EPSILON
            if new_grad >= 0.0:
                return x0 + dists[n]
            grad = new_grad
            last_semi_grad = semi_grad
        if grad - last_semi_grad > 0.0:
            return x0 + dists[-1]
        return x0 + dists[-1] + MAPELinearRegression.EPSILON


if __name__ == "__main__":
    N = 21
    # X = np.random.randint(10, 80, N).reshape(N, 1)
    # y = np.dot(X, np.array([0.8]).reshape(1, 1)).reshape(N) + 1 + 5 * np.random.randn(N)
    # y = np.maximum(y, 5)
    X = np.ones((N, 1))
    y = (np.random.randn(N) * 20 + 100).reshape(N)

    import sklearn.linear_model as skl
    lr = skl.LinearRegression()
    lr.fit(X, y)
    lr_results = lr.predict(X)
    lr.intercept_ *= 0.96

    regressor = MAPELinearRegression(True)
    regressor.fit(X, y)
    results = regressor.predict(X)

    print("linear regression")
    print("w", lr.coef_)
    print("b", lr.intercept_)
    print("MAPE", np.sum(np.abs(lr_results-y)/y)/N)
    print("RMSE", np.sqrt(np.dot(lr_results-y, lr_results-y)/N))

    print("min-MAPE")
    print("w", regressor.coef_)
    print("b", regressor.intercept_)
    print("MAPE", np.sum(np.abs(results-y)/y)/N)
    print("RMSE", np.sqrt(np.dot(results-y, results-y)/N))

