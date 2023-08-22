from Layers import Layer
import numpy as np

class Dense_idea(Layer) :
    def __init__ (self, output_shape : int = 2,input_shape : tuple = (100,), ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(*(input_shape[-1],output_shape))
        self.bias = np.random.randn(*(1,output_shape))
        self.i = 0
    def initialize (self,input_shape):
        self.input_shape = input_shape
        self.weights = np.random.randn(*(input_shape[-1],self.output_shape))
        self.bias = np.random.randn(*(1,self.output_shape)) 
    def forward (self, input : np.ndarray) -> np.ndarray:
        self.i += 1
        if (self.i == 100):
            pass
        self.input = input
        somme = np.sum(np.abs(self.weights), axis=0)
        weights = np.sqrt(self.weights**2)/somme
        output = input.dot(weights) + self.bias
        return output 
    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) -> np.ndarray:
        input_gradient = output.dot(self.weights.T)
        somme = np.sum(np.abs(self.weights), axis=0, keepdims=True)
        weights_gradient =  (self.input/somme).dot(output)
        self.weights -= learning_rate * weights_gradient
        bias_gradient = np.sum(output, axis=0)
        self.bias -= learning_rate * bias_gradient

        return input_gradient