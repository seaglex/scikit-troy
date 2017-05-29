import numpy as np


class MAPELinearRegression(object):
    EPSILON = 1e-6
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        return

    def predict(self, X):
        results = np.dot(X, self.coef_) + self.intercept_
        return results.squeeze()

    def fit(self, X, y):
        num, dim = X.shape
        y = y.reshape(num, 1)
        w = np.zeros((dim, 1))
        b = 0
        for itr in range(1000):
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
            semi_grads = - 1 / y
            new_b = self.get_optimal(b, errors, semi_grads)
            print("itr {0}, b: loss {1}({2}=>{3})".format(itr, loss, b, new_b))
            b = new_b
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
        semi_grads = semi_grads[indexes]
        if not len(semi_grads):
            return x0
        # minimize x, all positive errors ( assume smaller x, larger error)
        grad += np.sum(semi_grads)
        if grad >= 0:
            return x0 + dists[0] - MAPELinearRegression.EPSILON
        last_semi_grad = 0.0
        for n, semi_grad in enumerate(semi_grads):
            if grad - last_semi_grad >= 0:
                return x0 + dists[n-1] + MAPELinearRegression.EPSILON
            new_grad = grad - semi_grad - last_semi_grad
            if new_grad > 0:
                return x0 + dists[n] - MAPELinearRegression.EPSILON
            if new_grad == 0:
                return x0 + dists[n]
            grad = new_grad
            last_semi_grad = semi_grad
        return x0 + dists[-1] + MAPELinearRegression.EPSILON


if __name__ == "__main__":
    N = 20
    X = np.random.randint(10, 80, N).reshape(N, 1)
    y = np.dot(X, np.array([0.8]).reshape(1, 1)).reshape(N) + 5 + 5 * np.random.randn(N)
    y = np.max([y, 5 * np.ones(N)], axis=0)

    import sklearn.linear_model as skl
    lr = skl.LinearRegression()
    lr.fit(X, y)
    lr_results = lr.predict(X)

    regressor = MAPELinearRegression()
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

    print(np.array([y, results, lr_results]))
