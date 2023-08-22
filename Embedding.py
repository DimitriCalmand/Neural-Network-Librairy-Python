from Layers import Layer
import numpy as np

class Embedding(Layer):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.rand(vocab_size, embedding_dim)
    def initialize (self,input_shape):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        self.embeddings = self.weights[inputs]
        return self.embeddings

    def backward(self, grad_output, learning_rate):
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, self.inputs, grad_output)
        self.weights -= learning_rate * grad_weights
