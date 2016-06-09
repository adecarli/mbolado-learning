import numpy as np

class LinearRegression(object):
    def __init__(self, features, alpha):
        self.features = features
        self.alpha    = alpha

        self.weights  = np.random.random( [features] ) / 100
        self.bias     = np.random.random( [1] ) / 100

    def fit(self, X, Y):
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

    for i in range(1000):
        lr.fit(X, Y)

    print("")
    print(lr.weights)
    print(lr.bias)




if __name__ == '__main__': main()
