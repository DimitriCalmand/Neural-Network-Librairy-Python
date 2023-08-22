import numpy as np
class Optimizer:
    def __init__(self, name, params):
        self.name = name
        self.params = params
    def select(self, weights):
        if self.name == "Adam":
            return Adam(weights, self.params)
        elif self.name == "SGD":
            return SGD(weights, self.params)

class Adam():
    def __init__(self, weights, params):
        if "beta1" in params:
            self.beta1 = params["beta1"]
        else :
            self.beta1 = 0.9

        if "beta2" in params:
            self.beta2 = params["beta2"]
        else :
            self.beta2 = 0.999

        if "epsilon" in params:
            self.epsilon = params["epsilon"]
        else :
            self.epsilon = 1e-8
        
        self.weights = weights
        self.iteration = 0
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
    def optimize(self, gradient, learning_rate = 0.01):
        self.iteration += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)
        update = learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.weights -=  update
class SGD():
    def __init__(self, weights, params):
        self.weights = weights
    def optimize(self, gradient, learning_rate = 0.01):
        self.weights -= gradient*learning_rate
