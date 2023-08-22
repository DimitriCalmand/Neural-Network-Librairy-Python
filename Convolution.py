from audioop import bias
from Layers import Layer
import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
from keras import layers
from keras import models

class Conv2d (Layer) :
    def __init__ (self,filter : int,input_shape : tuple = (64,64,3), kernel_size : int = 3, padding :str = "same") -> None :
        self.input_shape = input_shape # width, height, chanel
        self.filter = filter
        self.kernel_size = kernel_size
        self.kernel_shape= (input_shape[-1],kernel_size,kernel_size,filter)
        self.bias = np.zeros((filter,))
        self.weights = np.random.randint(0,2,size = self.kernel_shape) -0.5  # (5,1,800,800)
        self.padding = padding
    def initialize (self,input_shape):
        self.input_shape = input_shape # width, height, chanel
        self.weights = np.random.randint(0,2,size = self.kernel_shape) -0.5  # (5,1,800,800)

    def forward (self, input) -> np.ndarray:
        self.input = input
        output = np.zeros((input.shape[0],)+self.bias.shape)
        output = np.array(layers.Conv2D(3,(3,3), input_shape = input.shape[1:]))
        for batch in range(input.shape[0]) :
            tps = self.bias.copy()
            for i in range (self.filter) : 
                for j in range (self.kernel_shape[0]) : # Chanel (rgb, gray ...)
                    tps[:,:,i] += signal.convolve2d(input[batch,:,:,j],self.weights[j,:,:,i], "valid")
            output[batch] = tps

        return output

    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) :
        input_gradient = np.zeros(self.input.shape)
        db = np.zeros(self.bias.shape)
        for batch in range(self.input.shape[0]):
            kernel_tps = self.weights.copy()
            for i in range(self.filter):
                for j in range (self.kernel_shape[0]) : # Chanel (rgb, gray ...)
                    self.weights[j,:,:,i] -= learning_rate * signal.convolve2d(self.input[batch,:,:,j],output[batch,:,:,i],"valid")
                    input_gradient[batch,:,:,j] += signal.convolve2d(output[batch,:,:,i],kernel_tps[j,:,:,i],"full")
                db += np.sum(output[batch,:,:,i])
        self.bias -= learning_rate * db
        return input_gradient





        
    