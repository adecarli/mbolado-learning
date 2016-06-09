import numpy as np

class LinearRegression(object):
    def __init__(self, features, alpha, eps = 1e-9):
        self.features = features
        self.alpha    = alpha
        self.eps      = eps

        self.weights  = np.random.random( [features] ) / 100
        self.bias     = np.random.random( [1] ) / 100

    def fit(self, X, Y, **kwargs):
        # X [samples, features]
        # Y [samples]

        if 'until_converge' in kwargs:
            until_converge = kwargs['until_converge']
        else:
            until_converge = False

        while True:
            h = self.pre_fit(X, Y)

            if self.J(h, Y) < self.eps: break
            if not until_converge: break

    def pre_fit(self, X, Y):
        samples = len(X)
        hs = []

        for i in range(samples):
            x = X[i]
            y = Y[i]

            h = self.h_t(x)
            hs.append(h)

            for j in range(self.features):
                self.weights[j] = self.weights[j] + self.alpha * (y - h) * x[j]

            self.bias = self.bias + self.alpha * (y - h)

        return np.array(hs)

    def J(self, h, y):
        return np.sum((h - y) ** 2)/2

    def predict(self, X):
        return np.array( [
            self.h_t(X[i])
            for i in range(len(X))
        ] )

        # print X[0]
        # return np.fromfunction(
        #     lambda i: self.h_t(X[i]),
        #     [len(X)],
        #     dtype=int
        # )

    def h_t(self, x):
        return self.bias + np.dot(self.weights, x)

def main():
    lr = LinearRegression(1, 0.05)

    X = np.array( [
        [1],
        [2]
    ] )
    Y = np.array( [
        [1],
        [2]
    ] )

    X_test = np.array( [
        [3],
        [4]
    ] )

    lr.fit(X, Y, until_converge=True)

    print(lr.predict(X_test))



if __name__ == '__main__': main()
