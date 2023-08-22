import numpy as np


class Loss :
    def __init__ (self, loss, loss_prime) :
        self.loss = loss
        self.loss_prime = loss_prime
    def forward(self,y_true, y_pred):
        return self.loss(y_true,y_pred)
    def backprop (self, y_true, y_pred):
        return self.loss_prime( y_true, y_pred)
class Mse (Loss) :
    def __init__ (self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))
        def mse_prime(y_true, y_pred):
            return (-2 * (y_true - y_pred) / y_pred.shape[0])
        super().__init__(mse,mse_prime)
class BinaryCrossEntropy (Loss) :
    def __init__ (self):      
        def binary_cross_entropy(y_true, y_pred):
            y_true = np.clip(y_true, 1e-9, 1 - 1e-9)
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
            return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()#.mean(axis=1).mean(axis=1).mean(axis=0)
        def binary_cross_entropy_prime(y_true, y_pred):
            y_true = np.clip(y_true, 1e-9, 1 - 1e-9)
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
            return (y_pred - y_true) / y_pred.shape[0] 
        super().__init__(binary_cross_entropy,binary_cross_entropy_prime)
class CrossEntropy (Loss) :
    pass

