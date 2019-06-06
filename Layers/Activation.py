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
        