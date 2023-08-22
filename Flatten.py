import numpy as np 
from Layers import Layer
class Flatten(Layer):
    def __init__ (self, input_shape : tuple = (34,34,3)) -> None : 
        self.input_shape = input_shape 
    def initialize (self,input_shape):
        self.input_shape = input_shape
    def forward (self, input : np.ndarray) -> np.ndarray:
        self.input_shape = input.shape
        self.max = np.max(np.abs(input))
        return input.reshape((input.shape[0],-1))
    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) -> np.ndarray:
        return output.reshape(self.input_shape) 
