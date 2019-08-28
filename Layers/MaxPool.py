import numpy as np
from numba import jit

@jit(nopython=True)
def convolve(batch_size,stride,kernel_size,Z_channels,Z_height,Z_width,Z,X):
    for n in range(batch_size):
        for c in range(Z_channels):
            for h in range(Z_height):
                for w in range(Z_width):
                    Z[n,h,w,c] += np.max(
                        X[
                            n,
                            h*stride[0]:h*stride[0]+kernel_size[0], 
                            w*stride[1]:w*stride[1]+kernel_size[1],
                            c
                        ]
                    )
    return Z

@jit(nopython=True)
def convolve_back(batch_size,stride,kernel_size,Z_channels,Z_height,Z_width,gradients,X,X_grad):
    def unravel_index(argmax,size):
        _,width = size
        row_id = argmax//width
        column_id = argmax%width
        return (row_id,column_id)
    
    for n in range(batch_size):
        for c in range(Z_channels):
            for h in range(Z_height):
                for w in range(Z_width):
                    binary_mask = np.zeros(kernel_size)
                    max_index = unravel_index(
                        np.argmax(
                            X[
                                n,
                                h*stride[0]:h*stride[0]+kernel_size[0],
                                w*stride[1]:w*stride[1]+kernel_size[1],
                                c
                            ]
                        ),
                        kernel_size
                    )
                    binary_mask[max_index[0],max_index[1]] += 1
                    X_grad[
                        n,
                        h*stride[0]:h*stride[0]+kernel_size[0],
                        w*stride[1]:w*stride[1]+kernel_size[1],
                        c
                    ] += binary_mask * gradients[n,h,w,c]
    return X_grad

class MaxPool(object):
    "Instantiate a max pool layer."
    def __init__(self, kernel_size, stride=None):
        try:
            int(kernel_size)
            kernel_size = (kernel_size,kernel_size)
        except:
            pass
        
        if stride:
            try:
                int(stride)
                stride = (stride,stride)
            except:
                pass
        else:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride

    def _init_weights(self,previous_layer):
        height,width,channels = previous_layer.output_height,previous_layer.output_width,previous_layer.filters
        self.output_height = int(1+(height - self.kernel_size[0])/self.stride[0])
        self.output_width = int(1+(width - self.kernel_size[1])/self.stride[1])
        self.filters = channels

    def _forward_pass(self,X):            
        batch_size,_,_,Z_channels = X.shape
        Z_height = int(1 + (X.shape[1] - self.kernel_size[0])/self.stride[0])
        Z_width = int(1 + (X.shape[2] - self.kernel_size[0])/self.stride[1])
        Z = np.zeros((batch_size,Z_height,Z_width,Z_channels))
        Z = convolve(batch_size,self.stride,self.kernel_size,Z_channels,Z_height,Z_width,Z,X)
        self.X = X
        return Z
    
    def _backward_pass(self,gradients):
        batch_size, Z_height, Z_width, Z_channels = gradients.shape
        X_grad = np.zeros_like(self.X)    
        X_grad = convolve_back(batch_size,self.stride,self.kernel_size,Z_channels,Z_height,Z_width,gradients,self.X,X_grad)    
        return X_grad