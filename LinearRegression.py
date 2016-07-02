#! /usr/bin/env python
"""Linear Regression

Implements a linear regression.
"""

import numpy as np

__author__      =   "Marcel Santana, Vitor Castelo Branco"
__copyright__   =   ""
__credits__     =   ["Marcel Santana", "Vitor Castelo Branco"]

__license__     =   ""
__version__     =   "0.1"
__maintainer__  =   "Marcel Santana, Vitor Castelo Branco"
__email__       =   "vtcb@cin.ufpe.br"
__status__      =   "Development"

class LinearRegression(object):
    """Linear Regression"""

    def __init__(self, features, alpha, eps = 1e-9):
        """Returns a LinearRegression object

    features: Number of features in the input
    alpha:    Learning rate
    eps:      Converging precision (optional)


    weights: Network parameter
    bias:    Network parameter"""

        self.features = features
        self.alpha    = alpha
        self.eps      = eps

        self.weights  = np.random.random( [features] ) / 100
        self.bias     = np.random.random( [1] ) / 100

    def fit(self, X, Y, **kwargs):
        """Train the network

    Trains the network using the Gradient Descente algorithm possibly several times

    X:              input, numpy.array of shape [samples, features]
    Y:              correct output, numpy.array of shape [samples]
    until_converge: whether it should repeat the training algorithm until the precision is at least eps (optional)"""
        steps = 0

        if 'until_converge' in kwargs:
            until_converge = kwargs['until_converge']
        else:
            until_converge = False

        while True:
            h = self.pre_fit(X, Y)

            if not until_converge: break
            if self.J(h, Y) < self.eps: break
            steps = steps + 1
            if steps > 10000: break

    def pre_fit(self, X, Y):
        """Train the network (once)

    Trains the network using the Gradient Descente algorithm

    X:              input, numpy.array of shape [samples, features]
    Y:              correct output, numpy.array of shape [samples]"""
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
        """Error function"""
        return np.sum((h - y) ** 2)/2

    def predict(self, X):
        """Predicts the output for given input

    Predicts the output for the given input based no the already trained network (It should be, you know...)

    X: input, numpy.array of shape [samples, features]
    """
        return np.array( [
            self.h_t(X[i])
            for i in range(len(X))
        ] )

    def h_t(self, x):
        """Predicts the output for one sample

    X: input, numpy.array of shape [features]"""
        return self.bias + np.dot(self.weights, x)

def main():
    lr = LinearRegression(2, 0.05, 1e-9)

    X = np.array( [
        [1, 2],
        [2, 4],
        [-1, 8]
    ] )
    Y = np.array( [
        [3],
        [6],
        [7]
    ] )

    X_test = np.array( [
        [3, 5],
        [4, 1]
    ] )

    lr.fit(X, Y, until_converge=True)

    print(lr.predict(X_test))



if __name__ == '__main__': main()
