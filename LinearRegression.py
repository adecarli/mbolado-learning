import numpy as np

class LinearRegression(object):
    def __init__(self, features, alpha):
        self.features = features
        self.alpha    = alpha

        self.weights  = np.random.random( [features] ) / 100
        self.bias     = np.random.random( [1] ) / 100

    def fit(self, X, Y, **kwargs):
        # X [samples, features]
        # Y [samples]

        samples = len(X)

        for i in range(samples):
            x = X[i]
            y = Y[i]

            h = self.h_t(x)
            for j in range(self.features):
                self.weights[j] = self.weights[j] + self.alpha * (y - h) * x[j]

            self.bias = self.bias + self.alpha * (y - h)

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

    for i in range(1000):
        lr.fit(X, Y)

    print(lr.predict(X_test))

    print("")
    print(lr.weights)
    print(lr.bias)




if __name__ == '__main__': main()
