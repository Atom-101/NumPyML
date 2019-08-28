import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import numpy as np
from Losses.Losses import *

loss_dict = {
    'mse': MeanSquaredError,
    'binary_cross_entropy' : BinaryCrossEntropy
    #cat crossentropy with logits
}

class Model(object):
    def __init__(self,input_shape):
        "Initializes an empty model. A model is a sequential list of layers"
        self.input_shape = input_shape
        self.layer_graph = []
        pass

    def add(self, obj):
        "Adds a layer to the model graph"
        if not self.layer_graph:
            obj._init_weights(input_shape=self.input_shape)
        else:
            obj._init_weights(previous_layer=self.layer_graph[-1])
        
        self.layer_graph.append(obj)

    def train(self, learning_rate, dataset, num_epochs, loss_fn):
        "Train model with given parameters"
        num_iters = int(dataset.length/dataset.batch_size)
        # For softmax with cross entropy, it is faster to use a linear activation in the final layer.
        # The loss function adds automatically does the softmax calculation """
        if loss_fn != 'cross_entropy_with_softmax':
            loss_fn,loss_fn_derivative = loss_dict[loss_fn].loss,loss_dict[loss_fn].loss_derivative
        else:
            loss_fn_derivative = loss_dict[loss_fn]
            loss_fn = None
        for _ in range(num_epochs):
            loss_log=[]
            for _ in range(num_iters):
                #Fwd pass
                X_train,y_train = dataset.next()
                out = self._forward_pass(X_train,True)
                #Compute loss
                if loss_fn:
                    loss = loss_fn(out,y_train)
                    print(loss)
                    loss_log.append(loss)
                    # @todo: write plot code

                #Backwd pass
                self._backward_pass(loss_fn_derivative,out,y_train)
                
                #Update params
                self._update_parameters(learning_rate)

    def _forward_pass(self,data,training):
        "Forward pass data through the model"
        for layer in self.layer_graph:
            if type(layer).__name__ == 'BatchNorm':
                data = layer._forward_pass(data,training)
            else:
                data = layer._forward_pass(data)
        return data

    def _backward_pass(self,loss_fn_derivative,out,y_train):
        "Backpropagate the loss through the model"
        # Gradient of final output wrt loss func
        grad = loss_fn_derivative(out,y_train)
        # for layer in reversed(self.layer_graph[1:]):
        for layer in reversed(self.layer_graph):
            grad = layer._backward_pass(grad)

    def _update_parameters(self,lr):
        "Update model parameters"
        for layer in self.layer_graph:
            if (type(layer).__name__ == 'Dense' or 
                    type(layer).__name__ == 'Conv'):
                layer.weights -= lr*layer.weights_grad
                layer.bias -= lr*layer.bias_grad
            elif type(layer).__name__ == 'BatchNorm':
                layer.gamma -= lr*layer.gamma_grad
                layer.beta -= lr*layer.beta_grad
        
    def predict(self):
        "Predict using trained model. To be implemented"
        raise NotImplementedError('Predict is not yet implemented')


        

