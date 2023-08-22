from Layers import Layer
import numpy as np
def random_debug(shape):
    val = 1
    for i in range (len(shape)):
        val *= shape[i]
    l = []
    for i in range(val):
        l.append(i/100)
    # return np.array(l).reshape(shape)
    return np.random.randn(*shape)/3
    # return np.ones(shape)
class Dense(Layer) :
    def __init__ (self, output_shape : int = 2,input_shape : tuple = (100,), ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(*(input_shape[-1],output_shape))
        self.bias = np.random.randn(*(1,output_shape))
    def initialize (self,input_shape):
        self.input_shape = input_shape
        self.weights = random_debug((input_shape[-1],self.output_shape))#np.ones((input_shape[-1],self.output_shape))
        # self.weights = np.random.randn(*(input_shape[-1],self.output_shape))/3
        # self.bias = np.random.randn(*(1,self.output_shape))
        self.bias = np.zeros((1,self.output_shape))
    def forward (self, input : np.ndarray) -> np.ndarray:
        # print(self.bias)
        self.input = input
        output = input.dot(self.weights) + self.bias
        return output 
    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) -> np.ndarray:
        if (len(self.input.shape) == 3):
            input_gradient = np.zeros(self.input.shape)
            for t in reversed(range(self.input.shape[1])):
                c = np.mean(self.input)
                q = output[:,t,:].dot(self.weights.T)
                input_gradient[:,t,:] = output[:,t,:].dot(self.weights.T)
                self.weights -= learning_rate * self.input[:,t,:].T.dot(output[:,t,:])
                a = output[:,t,:]
                self.bias -= learning_rate * np.sum(output[:,t,:], axis=0, keepdims=True)
            return input_gradient
        
        input_gradient = output.dot(self.weights.T)
        weights_gradient =  self.input.T.dot(output)
        # print("weights = \n",weights_gradient)
        self.weights -= learning_rate * weights_gradient
        bias_gradient = np.sum(output, axis=0)
        # print(bias_gradient) 
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient


