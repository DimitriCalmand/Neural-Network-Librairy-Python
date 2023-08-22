import numpy as np
from Layers import Layer
class Dropout(Layer):
    def __init__(self, rate: float, input_shape: tuple = (100, )):
        self.rate = rate
        self.input_shape = input_shape
        self.mask = None
    
    def initialize(self, input_shape, optimizer):
        self.input_shape = input_shape
    
    def forward(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) / (1 - self.rate)
            output = input * self.mask
        else:
            output = input
        return output
    
    def backprop(self, output: np.ndarray, learning_rate: float = 1e-2) -> np.ndarray:
        input_gradient = output * self.mask
        return input_gradient
