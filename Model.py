import numpy as np
from Convolution import Conv2d 
from Layers import Layer
from Loss import Loss
from Activations import Activation, Relu
from tqdm import tqdm
from NN_idea import Dense_idea
from time import time
import matplotlib.pyplot as plt

from Pooling import Pooling

class Model : 
    def __init__ (self) -> None :
        self.layers = list()
    def add (self, layer : Layer) :
        self.layers.append(layer)
    def compile (self, loss_function : Loss) :
        self.loss_function = loss_function
        shape = self.layers[0].input_shape
        # X = np.random.randn(*(1,)+shape)
        X = np.ones((1,)+shape)
        for layer in self.layers :
            layer.initialize(shape)  
            X = layer.forward(X)
            shape = X.shape[1:]



    def predict(self, X : np.ndarray):
        for layer in self.layers:
            temps = time()            
            X = layer.forward(X)
        return X
    def accuracy(self,prediction : np.ndarray, Y : np.ndarray) :
        """ Return the model performance as a float number between 0 and 1."""
        res = 0
        if prediction.shape[-1] > 1 : # Not sigmoid
            pred = np.argmax(prediction, axis=-1)
            true = np.argmax(Y, axis=-1)
            good = pred == true
            res = np.sum(good)
        else :
            pred = prediction > 0.5
            true = Y > 0.5
            good = pred == true
            res = np.sum(good)
        return res/np.size(pred)


    def fit (self, train : tuple ,test : tuple ,epoch, learning_rate, batch_size = 100 ):
        x_train, y_train = train 
        x_test, y_test = test
        nb_print = epoch//10
        nb_print = max(1,nb_print)
        tps = time()
        for i in range (epoch) :            
            for batch in range(x_train.shape[0]//batch_size ) :
                X = x_train[batch*batch_size:batch_size*(batch+1)]
                Y = y_train[batch*batch_size:batch_size*(batch+1)]
                prediction = self.predict(X)
                loss = self.loss_function.forward(Y, prediction)
                output = self.loss_function.backprop(Y,prediction)
                #Backpropagation
                for layer in reversed(self.layers) :                       
                    output = layer.backprop(output, learning_rate = learning_rate)
            if (i%nb_print==nb_print-1):
                accuracy =  self.accuracy(self.predict(x_test),y_test)
                print(f"epoch {i}/{epoch} | time : {round(time()-tps,3)} secondes | acc = {accuracy} | loss = {round(loss,4)} |")
                tps = time()
                pass
    


