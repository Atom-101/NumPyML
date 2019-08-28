import numpy as np

from Activations.standard_activations import *

activations_dict = {
    'relu': (relu,relu_backward), 
    'sigmoid': (sigmoid,sigmoid_backward),
    'leaky_relu': (leaky_relu,leaky_relu_backward)
    #tanh
}

class Activation(object):
    def __init__(self,activation):
        self.activation,self.activation_derivative = activations_dict[activation]
    
    def _init_weights(self,previous_layer=None):
        try:
            self.units = previous_layer.units
        except:
            self.output_height,self.output_width,self.filters = previous_layer.output_height,previous_layer.output_width,previous_layer.filters

    def _forward_pass(self,X):
        self.X = X
        return self.activation(X)

    def _backward_pass(self,gradients):
        return self.activation_derivative(gradients,self.X)
