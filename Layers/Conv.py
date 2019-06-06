import numpy as np
import math
from Activations.standard_activations import *

activations_dict = {
    'relu': (relu,relu_backward), 
    'sigmoid': (sigmoid,sigmoid_backward)
    #lrelu
    #tanh
}

class Conv(object):
    def __init__(self,kernel_size,filters,stride,padding,activation):
        # Follows (Filters, Height, Width, Channels) convention
        try:
            int(kernel_size)
            kernel_size = (kernel_size,kernel_size)
        except:
            pass
        try:
            int(stride)
            stride = (stride,stride)
        except:
            pass        
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.activation, self.activation_derivative = activations_dict[activation]
        
        
    def _init_weights(self,previous_layer=None,input_shape=None):        
        if input_shape:
            height,width,channels = input_shape
        else:
            height,width,channels = previous_layer.output_height,previous_layer.output_width,previous_layer.filters

        if self.padding == 'same':
            padh = math.ceil((self.stride[0]-1)*height - self.stride[0] + self.kernel_size[0])
            padw = math.ceil((self.stride[1]-1)*width - self.stride[1] + self.kernel_size[1])
        else:
            padh,padw = 0,0
        self.output_height = int((height - self.kernel_size[0] + padh)/self.stride[0] + 1)
        self.output_width = int((width - self.kernel_size[1] + padw)/self.stride[1] + 1)
        
        self.weights = np.random.normal(
            loc=0,
            scale=0.1,
            size=(self.filters,self.kernel_size[0],self.kernel_size[1],channels)
        )
        self.bias = np.zeros((self.filters))
        

    def _forward_pass(self,X):
        # @todo: handle padding
        X,pad = self._pad(X)
        self.pad = pad

        batch_size = X.shape[0]
        Z_height = int(1 + (X.shape[1] - self.weights.shape[1])/self.stride[0])
        Z_width = int(1 + (X.shape[2] - self.weights.shape[2])/self.stride[1])
        Z = np.zeros((batch_size,Z_height,Z_width,self.filters))
        
        for n in range(batch_size):
            for f in range(self.filters):
                for h in range(Z_height):
                    for w in range(Z_width):
                        Z[n,h,w,f] = np.sum(
                            X[
                                n, 
                                h*self.stride[0]:h*self.stride[0]+self.kernel_size[0], 
                                w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                                :
                            ] *
                            self.weights[f]
                        )
                        Z[n,h,w,f] += self.bias[f]
        self.X = X
        self.Z = Z

        return self.activation(Z)


    def _backward_pass(self, gradients):
        gradients = self.activation_derivative(gradients, self.Z)
        batch_size, Z_height, Z_width = gradients.shape[:3] 
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        X_grad_padded = np.zeros_like(self.X,dtype=np.float32)
        
        for n in range(batch_size):
            for f in range(self.filters):
                for h in range(Z_height):
                    for w in range(Z_width):
                        self.weights_grad[f] += np.multiply( 
                            self.X[
                                n,
                                h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                                w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                                :
                            ],
                            gradients[n,h,w,f]
                        )
                        self.bias_grad[f] += gradients[n,h,w,f]
                        X_grad_padded[
                            n,
                            h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                            w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                            :
                        ] += self.weights[f] * gradients[n,h,w,f]
        
        X_grad = X_grad_padded[
            :,
            self.pad[0]:-self.pad[1],
            self.pad[2]:-self.pad[3],
            :
        ]
        return X_grad

    def _pad(self, X):
        def _pad_util(height,width):
            pad = [0,0,0,0]
            if self.padding == 'valid':
                return pad
            pad_h = math.ceil((self.stride[0]-1)*height - self.stride[0] + self.kernel_size[0])
            pad_w = math.ceil((self.stride[1]-1)*width - self.stride[1] + self.kernel_size[1])
            pad[0] = pad_h//2
            pad[1] = pad_h - pad[0]
            pad[2] = pad_w//2
            pad[3] = pad_w - pad[2]
            return pad
        
        _,height,width,_ = X.shape
        pad = _pad_util(height,width)
        X = np.pad(X, ((0,0),pad[:2],pad[2:],(0,0)), 'constant')
        return X,pad      
        
        