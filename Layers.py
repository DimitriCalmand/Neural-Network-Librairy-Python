import numpy as np
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input : np.ndarray):
        pass
    def backprop(self, output : np.ndarray, learning_rate : float = 1e-2):
        pass
    def initialize (self,input_shape):
        pass