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

    def _forward_pass(self,X):
        batch_size,Z_channels = X.shape[0],X.shape[3]
        Z_height = int(1 + (X.shape[1] - self.kernel_size[0])/self.stride)
        Z_width = int(1 + (X.shape[2] - self.kernel_size[0])/self.stride)
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
                        binary_mask = np.zeros((self.kernel_size[0],self.kernel_size[1]))
                        max_index = np.unravel_index(
                            np.argmax(
                                self.X[
                                    n,
                                    h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                                    w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                                    c
                                ]
                            )
                        )
                        binary_mask[max_index[0],max_index[1]] += 1
                        X_grad[
                            n,
                            h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                            w*self.stride[1]:w*self.stride[1]+self.kernel_size[1],
                            c
                        ] += binary_mask * gradients[n,h,w,c]