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

    def predict(self, feature):
        w = {}
        wTx = 0
        for i in feature:
            sign = 0 
            if self.z[i] < 0:
                sign = -1 
            else: 
                sign = 1
            if abs(self.z[i]) <= self.L1:
                self.w[i] = 0.
            else:
                self.w[i] = (-((self.beta + sqrt(self.n[i])) / self.alpha + self.L2) ** -1) * (self.z[i] - sign * self.L1)
            wTx += self.w[i]
        # ograniczenie funkcji exp, aby nie osiągała za dużych wartości (wtedy jej odwrotność jest bardzo bliska zeru)
        return 1 / (1 + exp(-max(min(wTx, 35), -35)))

    def update_model(self, feature, probability, clicked):
        g = probability - clicked
        for i in feature:
            sigma = (sqrt(self.n[i] + g ** 2) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g ** 2
