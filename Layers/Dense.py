import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import numpy as np
from Activations.standard_activations import *

activations_dict = {
    'relu': (relu,relu_backward), 
    'sigmoid': (sigmoid,sigmoid_backward),
    'linear': (linear,linear_backward)
    #tanh
}
class Dense(object):
    "Instantiate a dense layer"
    def __init__(self,units,activation):
        if activation not in activations_dict.keys():
            raise ValueError("Invalid activation type")
        self.units = units
        self.activation,self.activation_derivative = activations_dict[activation]
        # self.cache = []
        
    # Init performed by parent Model object
    def _init_weights(self,previous_layer=None,input_shape=None):        
        if hasattr(previous_layer,'weights') and previous_layer.weights.ndim != 2:
            raise ValueError("Invalid input shape for Dense layer."
             "Dense layer can only be preceded by Input, Dense or Flatten layers")    
        if input_shape:
            self.weights = np.random.normal(loc=0, scale=0.1, size=(input_shape,self.units))
        else:
            self.weights = np.random.normal(loc=0, scale=0.1, size=(previous_layer.units,self.units))       
        self.bias = np.zeros(self.units)
        # self.previous_layer = previous_layer

    def _forward_pass(self, X):
        Z = np.matmul(X,self.weights) + self.bias
        A = self.activation(Z)
        self.X = X
        self.Z = Z
        return A

    def _backward_pass(self, gradients):
        gradients = self.activation_derivative(gradients,self.Z)
        self.weights_grad = np.matmul(self.X.T,gradients)
        self.bias_grad = np.sum(gradients,axis=0)
        input_grad = np.matmul(gradients, self.weights.T)
        return input_grad


