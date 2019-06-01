import numpy as np
from Activations.standard_activations import *

activations_dict = {
    'relu': (relu,reluBackward), 
    'sigmoid': (sigmoid,sigmoidBackward)
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
        
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.activation, self.activation_derivative = activations_dict[activation]
        
        
    def _init_weights(self,previous_layer=None,input_shape=None):        
        if input_shape:
            channels = input_shape
        else:
            channels = previous_layer.filters

        
        self.weights = np.random.normal(
            loc=0,
            scale=0.1,
            size=(self.filters,self.kernel_size[0],self.kernel_size[1],channels)
        )
        self.bias = np.zeros((self.filters))

    def _forward_pass(self,X):
        # @todo: handle padding
        # X,pad = pad(X,self.padding)
        # self.padding = pad [left,up,right,down]

        batch_size = X.shape[0]
        Z_height = int(1 + (X.shape[1] - self.weights.shape[1])/self.stride)
        Z_width = int(1 + (X.shape[2] - self.weights.shape[2])/self.stride)
        Z = np.zeros((batch_size,Z_height,Z_width,self.filters))
        

        for n in range(batch_size):
            for f in range(self.filters):
                for h in range(Z_height):
                    for w in range(Z_width):
                        Z[n,h,w,f] = np.sum(
                            X[
                                n, 
                                h*self.stride:h*self.stride+self.kernel_size[0], 
                                w*self.stride:w*self.stride+self.kernel_size[1],
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
        X_grad_padded = np.zeros_like(self.X)
        
        for n in range(batch_size):
            for f in range(self.filters):
                for h in range(Z_height):
                    for w in range(Z_width):
                        self.weights_grad[f] += np.multiply( 
                            self.X[
                                n,
                                h*self.stride:h*self.stride+self.kernel_size[0],
                                w*self.stride:w*self.stride+self.kernel_size[1],
                                :
                            ],
                            gradients[n,h,w,f]
                        )
                        self.bias_grad[f] += gradients[n,h,w,f]
                        X_grad_padded[
                            n,
                            h*self.stride:h*self.stride+self.kernel_size[0],
                            w*self.stride:w*self.stride+self.kernel_size[1],
                            :
                        ] += self.weights[f] * gradients[n,h,w,f]
        
        X_grad = X_grad_padded[
            :,
            self.padding[0]:-(self.padding[2]+1),
            self.padding[1]:-(self.padding[3]+1),
            :
        ]
        return X_grad


        