from dbm import ndbm
import numpy as np
from Layers import Layer


class Activation (Layer) :
    def __init__ (self, activation, activation_prime) :
        self.activation = activation
        self.actication_prime = activation_prime
    def initialize (self,input_shape):
        self.input_shape = input_shape
    def forward(self, input: np.ndarray):
        return self.activation(input)
    def backprop(self, output: np.ndarray, learning_rate: float = 0.01):
        return self.actication_prime(output,learning_rate = learning_rate)

class Relu (Activation): 
    def __init__ (self) -> None:
        def relu (input : np.ndarray) -> np.ndarray:
            return np.maximum(0,input)
        def relu_prime (output : np.ndarray, learning_rate = 1e-2) -> np.ndarray:
            output = np.where(output <= 0, 0, output)
            output = np.where(output > 0, 1, output)
            return output
        super().__init__(relu,relu_prime)

class Sigmoid(Activation) :
    def __init__ (self) -> None:        
        def sigmoid(input : np.ndarray) -> np.ndarray:
            self.input = 1/(1+np.exp(-input))
            return self.input.copy()
        def sigmoid_prime (output : np.ndarray, learning_rate : float = 1e-2 ) -> np.ndarray:
            input_gradient = self.input*(1-self.input)#1/(1+np.exp(-self.input)) * (1 - 1/(1+np.exp(-self.input)))
            res =  output * input_gradient
            return res
        super().__init__(sigmoid,sigmoid_prime)

class Tanh(Activation) :
    def __init__ (self) -> None:        
        def tanh(input : np.ndarray) -> np.ndarray:
            self.input = np.tanh(input)
            return self.input
        def tanh_prime (output : np.ndarray, learning_rate : float = 1e-2 ) -> np.ndarray:
            input_gradient = output * (1 - self.input**2)
            return input_gradient
        super().__init__(tanh,tanh_prime)

class Softmax(Activation) :
    def __init__ (self) -> None:        
        def softmax(input : np.ndarray) -> np.ndarray:
            val = np.exp(input)
            a = np.sum(val, axis=-1, keepdims=True)
            self.input = val / np.sum(val, axis=-1, keepdims=True)
            return self.input.copy()
        def softmax_prime (output : np.ndarray, learning_rate : float = 1e-2 ) -> np.ndarray:
            return output * self.input * (1 - self.input)
        super().__init__(softmax,softmax_prime)




