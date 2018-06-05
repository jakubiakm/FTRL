from random import random
from math import sqrt, exp

class FTRL:
    def __init__(self, alpha, beta, L1, L2):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.n = [0.] * 2 ** 28  
        self.z = [0.] * 2 ** 28
        self.w = {}
        print("FTRL execution start")

    def _indices(self, x):
        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index


    def predict(self, x):
        w = {}
        wTx = 0
        for i in self._indices(x):
            sign = -1. if self.z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * self.z[i] <= self.L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * self.L1 - self.z[i]) / ((self.beta + sqrt(self.n[i])) / self.alpha + self.L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g
