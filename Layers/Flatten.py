import numpy as np

class Flatten(object):
    
    def _init_weights(self,previous_layer):
        height,width,channels = previous_layer.output_height,previous_layer.output_width,previous_layer.filters
        self.units = height*width*channels

    def _forward_pass(self, X):
        self.shape = X.shape
        return np.reshape(X,(self.shape[0],-1))

    def _backward_pass(self, gradients):
        return np.reshape(gradients,self.shape)


