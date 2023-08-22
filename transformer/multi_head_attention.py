import numpy as np
# from Layers import Layer
from head_attention import head_attention
class MultiHeadAttention():
    def __init__(self, num_heads = 8, d_model = 128, input_shape = (100,128)):
        self.num_heads = num_heads
        self.d_model = d_model
        self.input_shape = input_shape
        self.dK = d_model // num_heads
    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.WQ = np.random.randn(self.input_shape[1], self.d_model)
        self.WK = np.random.randn(self.input_shape[1], self.d_model)
        self.WV = np.random.randn(self.input_shape[1], self.d_model)
        self.WO = np.random.randn(self.d_model, self.input_shape[1])
        self.bQ = np.random.randn(1, self.d_model)
        self.bK = np.random.randn(1, self.d_model)
        self.bV = np.random.randn(1, self.d_model)
        self.bO = np.random.randn(1, self.input_shape[1])
    def forward(self, input):
        Q = input.dot(self.WQ) + self.bQ
        K = input.dot(self.WK) + self.bK
        V = input.dot(self.WV) + self.bV
        Q = Q.reshape((Q.shape[0], Q.shape[1], self.num_heads, self.dK))
        K = K.reshape((K.shape[0], K.shape[1], self.num_heads, self.dK))
        V = V.reshape((V.shape[0], V.shape[1], self.num_heads, self.dK))
        Q = np.transpose(Q, (2,1,0,3)).transpose((0,2,1,3))
        K = np.transpose(K, (2,1,0,3)).transpose((0,2,1,3))
        V = np.transpose(V, (2,1,0,3)).transpose((0,2,1,3))
        head = []
        for q, k, v in zip(Q, K, V):
            head.append(head_attention(q, k, v, self.dK))
        head = np.array(head)
        self.head = np.concatenate(head, axis = 2)        
        output = self.head.dot(self.WO) + self.bO
        return output

    def backprop(self, output, learning_rate = 1e-2):
        dloss_dO = output
        dloss_dWO = np.matmul(self.head.transpose((0,2,1)), dloss_dO).mean(axis = 0)
        dloss_dBO = dloss_dO.mean(axis = 0)
        self.WO -= learning_rate * dloss_dWO
        self.bO -= learning_rate * dloss_dBO

# test = MultiHeadAttention(d_model=256)
# test.initialize((4,256))
# x = np.random.randn(1,4,256)
# y = test.forward(x)
# print(y.shape)
# test.backprop(y)