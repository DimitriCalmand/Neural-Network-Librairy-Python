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

class Lstm(Layer) :
    def __init__ (self, input_shape = (14,1), hidden_units : int = 2, return_sequence = True):
        self.w_f, self.w_i, self.w_c, self.w_o = None, None, None, None
        self.b_f, self.b_i, self.b_c, self.b_o = None, None, None, None
        self.embedding_size = None
        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.return_sequence = return_sequence
    def initialize (self,input_shape):
        self.input_shape = input_shape
        self.embedding_size = input_shape[-1]
        # init with one 
        self.wf = random_debug((self.hidden_units+self.embedding_size, self.hidden_units))
        self.wi = random_debug((self.hidden_units+self.embedding_size, self.hidden_units))
        self.wc = random_debug((self.hidden_units+self.embedding_size, self.hidden_units))
        self.wo = random_debug((self.hidden_units+self.embedding_size, self.hidden_units))
        self.bf = np.zeros((1,self.hidden_units), dtype=np.float32)
        self.bi = np.zeros((1,self.hidden_units), dtype=np.float32)
        self.bc = np.zeros((1,self.hidden_units), dtype=np.float32)
        self.bo = np.zeros((1,self.hidden_units), dtype=np.float32)
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __sigmoid_prime(self, x):
        # return self.__sigmoid(x) * (1 - self.__sigmoid(x))
        return x*(1-x)
    def __cliping_gradient(self, gradient, threshold):
        return np.clip(gradient, -threshold, threshold)
    def __regularization(self, gradients, threshold):
        gradient_norm = np.linalg.norm(gradients)        
        if gradient_norm > threshold:
            regularized_gradient = gradients * (threshold / gradient_norm)
        else:
            regularized_gradient = gradients
        return regularized_gradient
    
    def forward (self, input : np.ndarray) -> np.ndarray:
        self.input = input
        shape = (input.shape[0], input.shape[1], self.hidden_units)
        self.h_out = np.zeros(shape)
        self.c_out = np.zeros(shape)
        self.ft_list = np.zeros((input.shape[1], input.shape[0] ,self.hidden_units))
        self.it_list = np.zeros((input.shape[1], input.shape[0] ,self.hidden_units))
        self.ct_list = np.zeros((input.shape[1], input.shape[0] ,self.hidden_units))
        self.ot_list = np.zeros((input.shape[1], input.shape[0] ,self.hidden_units))
        h_t_1 = np.zeros((self.h_out.shape[0], self.h_out.shape[-1]))
        self.concat_list = np.zeros((input.shape[1], input.shape[0] , self.hidden_units + self.embedding_size))
        for t in range(input.shape[1]):
            concat = np.hstack((input[:,t,:], h_t_1))

            ft = self.__sigmoid(concat.dot(self.wf) + self.bf)
            it = self.__sigmoid(concat.dot(self.wi) + self.bi)
            a = concat.dot(self.wc) + self.bc
            ct = np.tanh(concat.dot(self.wc) + self.bc)
            self.c_out[:,t,:] = ft * self.c_out[:,t-1,:] + it * ct
            ot = self.__sigmoid(concat.dot(self.wo) + self.bo)
            b = np.tanh(self.c_out[:,t,:])
            w = self.c_out[:,t,:]
            self.h_out[:,t,:] = ot * np.tanh(self.c_out[:,t,:])
            self.ft_list[t], self.it_list[t], self.ct_list[t], self.ot_list[t] = ft, it, ct, ot
            self.concat_list[t] = concat
            h_t_1 = self.h_out[:,t,:]
        a = np.mean(self.h_out)
        if self.return_sequence:
            return self.h_out
        return self.h_out[:,-1,:]
    
    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) -> np.ndarray:
        input_gradient = np.zeros(self.input.shape)
        input_gradient_1 = output
        dLoss_dWo = np.zeros(self.wo.shape)
        dLoss_dWi = np.zeros(self.wi.shape)
        dLoss_dWf = np.zeros(self.wf.shape)
        dLoss_dWc = np.zeros(self.wc.shape)
        dLoss_dBo = np.zeros(self.bo.shape)
        dLoss_dBi = np.zeros(self.bi.shape)
        dLoss_dBf = np.zeros(self.bf.shape)
        dLoss_dBc = np.zeros(self.bc.shape)
        regularization = 5    
        for t in reversed(range(self.h_out.shape[1])) :
            # h_out gradient
            if self.return_sequence :
                dLoss_dHout = output[:,t,:]
            else:
                dLoss_dHout = input_gradient_1
            # output gate gradient
            dLoss_dOt = dLoss_dHout * np.tanh(self.c_out[:,t,:])
            a = self.concat_list[t]
            b = self.__sigmoid_prime(self.ot_list[t])
            
            # cell state gradient
            a = self.c_out[:,t-1,:]
            itt = self.ot_list[t]
            b = np.tanh(self.c_out[:,t,:])
            c = dLoss_dHout * self.ot_list[t] #* (1 - np.tanh(self.c_out[:,t,:])**2)
            dLoss_dCout = dLoss_dHout * self.ot_list[t] * (1 - np.tanh(self.c_out[:,t,:])**2) * self.it_list[t]
            # input gate gradient
            dLoss_dIt = dLoss_dHout * self.ot_list[t] * (1 - np.tanh(self.c_out[:,t,:])**2) * self.ct_list[t]            
            # forget gate gradient
            dLoss_dFt = dLoss_dHout * self.ot_list[t] * (1 - np.tanh(self.c_out[:,t,:])**2) * self.c_out[:,t-1,:]
            
            if self.return_sequence or t==self.h_out.shape[1]-1:
                dLoss_dWc += self.concat_list[t].T.dot(dLoss_dCout * (1 - np.tanh(self.ct_list[t])**2))
                dLoss_dBc += np.sum(dLoss_dCout, axis = 0)

                dLoss_dWo += self.concat_list[t].T.dot(dLoss_dOt * self.__sigmoid_prime(self.ot_list[t]))
                dLoss_dBo += np.sum(dLoss_dOt, axis = 0) # sum a changer peut etre

                dLoss_dWi += self.concat_list[t].T.dot(dLoss_dIt * self.__sigmoid_prime(self.it_list[t]))
                dLoss_dBi += np.sum(dLoss_dIt, axis = 0)

                dLoss_dWf += self.concat_list[t].T.dot(dLoss_dFt * self.__sigmoid_prime(self.ft_list[t]))
                dLoss_dBf += np.sum(dLoss_dFt, axis = 0)
            # input gradient
            dLoss_dHout = dLoss_dHout.dot(self.wo.T) + dLoss_dCout.dot(self.wc.T) + dLoss_dIt.dot(self.wi.T) + dLoss_dFt.dot(self.wf.T)
            input_gradient[:,t,:] = dLoss_dHout[:,:self.embedding_size].copy()
            input_gradient_1 = self.__regularization(dLoss_dHout[:,self.embedding_size:].copy(), regularization)
            # update weightsprint

        self.wf -= learning_rate * dLoss_dWf    
        self.wi -= learning_rate * dLoss_dWi
        self.wc -= learning_rate * dLoss_dWc
        self.wo -= learning_rate * dLoss_dWo
        self.bf -= learning_rate * dLoss_dBf
        self.bi -= learning_rate * dLoss_dBi
        self.bc -= learning_rate * dLoss_dBc
        self.bo -= learning_rate * dLoss_dBo

        return input_gradient#self.__regularization(input_gradient, regularization)


    
            

        



