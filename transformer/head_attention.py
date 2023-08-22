import numpy as np

def head_attention(Q, K, V, dK):
    K = K.transpose((0,2,1))
    matmul1 = np.matmul(Q,K )
    scale = matmul1 / np.sqrt(dK)
    softmax = np.exp(scale) / np.sum(np.exp(scale))
    matmul2 = np.matmul(softmax, V)

    return matmul2

# q = np.random.randn(1,1,256)
# k = np.random.randn(1,4,256)     
# v = np.random.randn(1,4,256)
# dk = 256 // 8
# y = head_attention(q, k, v, dk)
# print(y.shape)