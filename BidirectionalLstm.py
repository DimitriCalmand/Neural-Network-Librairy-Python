from LSTM import Lstm
from Layers import Layer
import numpy as np
from Optimizer import Optimizer
# class BidirectionalLstm(Layer):
#     def __init__(self, input_shape=(14,1), hidden_units=2, return_sequences=True):
#         self.input_shape = input_shape
#         self.hidden_units = hidden_units
#         self.return_sequences = return_sequences
#         self.lstm_forward = Lstm(input_shape=input_shape, hidden_units=hidden_units, return_sequence=return_sequences)
#         self.lstm_backward = Lstm(input_shape=input_shape, hidden_units=hidden_units, return_sequence=return_sequences)

#     def initialize(self, input_shape, optimizer:Optimizer):
#         self.lstm_forward.initialize(input_shape, optimizer)
#         self.lstm_backward.initialize(input_shape, optimizer)

#     def forward(self, input:np.ndarray) -> np.ndarray:
#         input_backward = np.flip(input, axis=1)
#         output_forward = self.lstm_forward.forward(input)
#         output_backward = np.flip(self.lstm_backward.forward(input_backward), axis=1)
#         output = np.concatenate((output_forward, output_backward), axis=-1)
#         return output

#     def backprop(self, output:np.ndarray, learning_rate:float=1e-2) -> np.ndarray:
#         output_backward = np.flip(output, axis=1)
#         half_output_dim = output.shape[-1] // 2
#         if self.return_sequences:
#             gradient_forward = output[:,:,:half_output_dim]
#             gradient_backward = np.flip(output_backward[:,:,:half_output_dim], axis=1)
#         else:
#             gradient_forward = output[:,:half_output_dim]
#             gradient_backward = np.flip(output_backward[:,:half_output_dim], axis=1)

        
#         gradient_input_forward = self.lstm_forward.backprop(gradient_forward, learning_rate)
#         gradient_input_backward = np.flip(self.lstm_backward.backprop(gradient_backward, learning_rate), axis=1)
#         gradient_input = gradient_input_forward + gradient_input_backward
#         return gradient_input

