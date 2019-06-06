import numpy as np

class MaxPool(object):
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

        for n in range(batch_size):
            for c in range(Z_channels):
                for h in range(Z_height):
                    for w in range(Z_width):
                        Z[n,h,w,c] += np.max(
                            X[
                                n,
                                h*self.stride[0]:h*self.stride[0]+self.kernel_size[0], 
                                w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                                c
                            ]
                        )
        self.X = X
        return Z

    def _backward_pass(self,gradients):
        batch_size, Z_height, Z_width, Z_channels = gradients.shape
        X_grad = np.zeros_like(self.X)    

        for n in range(batch_size):
            for c in range(Z_channels):
                for h in range(Z_height):
                    for w in range(Z_width):
                        binary_mask = np.zeros(self.kernel_size)
                        max_index = np.unravel_index(
                            np.argmax(
                                self.X[
                                    n,
                                    h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                                    w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                                    c
                                ]
                            ),
                            self.kernel_size
                        )
                        binary_mask[max_index[0],max_index[1]] += 1
                        X_grad[
                            n,
                            h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                            w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                            c
                        ] += binary_mask * gradients[n,h,w,c]
        return X_grad