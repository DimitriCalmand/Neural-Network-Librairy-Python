import numpy as np
from scipy import signal
from Layers import Layer
from time import time
import cv2 
import matplotlib.pyplot  as plt 
class Pooling(Layer) :
    def __init__(self, input_shape : tuple = (64,64,128)) -> None :
        self.input_shape = input_shape # Example : (32,64,64) where 32-> filters, 64-> size
    def forward(self, input : np.ndarray) -> np.ndarray :
        self.input = input
        batch_size,height, width,filters = self.input.shape
        height,width = height//2,width//2
        output = np.zeros((batch_size,height,width,filters))
        for batch in range(batch_size) :      
            for h in range (height) :
                for w in range (width) :    
                    for i in range (filters): # Filters                
                        output[batch,h,w,i] = np.amax(input[batch,h*2:h*2+2,w*2:w*2+2,i])
        
        self.input_after_pool = output
        return output
    def initialize (self,input_shape):
        self.input_shape = input_shape
    def backprop (self, output : np.ndarray, learning_rate : float = 1e-2) :
        input_gradient = np.zeros(self.input.shape)
        batch_size,  height, width,filters = self.input.shape
        self.input_after_pool = np.pad(self.input_after_pool, (0, 1), 'constant', constant_values=(0, 0))
        output = np.pad(output, (0, 1), 'constant', constant_values=(0, 0))
        for batch in range(batch_size) :            
            for h in range (height) :
                for w in range (width) :
                    for i in range (filters) :
                        if self.input[batch,h,w,i] == self.input_after_pool[batch,h//2,w//2,i] :
                            input_gradient[batch,h,w,i] = output[batch,h//2,w//2,i]                          
        return input_gradient



# import cv2 
# import matplotlib.pyplot  as plt 

# image = cv2.resize(cv2.imread("ImageDimitri9.png",0),(128,128)).reshape((1,128,128,1))

# conv = Pooling(None)
# image = conv.forward(image)
# print(image.shape)

# image = conv.backprop(image).astype(int)
# plt.imshow(image[0,:,:,0],cmap = "gray")
# plt.show()



