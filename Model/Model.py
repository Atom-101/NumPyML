import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import numpy as np
from Losses.Losses import *

loss_dict = {
    'mse': (mean_squared_error,mean_squared_error_derivative)
    #bin crossentropy
    #cat crossentropy with logits
}

class Model(object):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.layer_graph = []
        pass

    def add(self, obj):
        '''
        Attributes:
        obj: layer object to be added to model graph
        '''        
        # if (not self.layer_graph 
        #         and type(obj).__name__ != 'Input'):
        #     raise AttributeError('Model should begin with input layer')
        
        # if type(obj).__name__ != 'Input':
        #     obj._init_weights(self.layer_graph[-1])

        if not self.layer_graph:
            obj._init_weights(input_shape=self.input_shape)
        else:
            obj._init_weights(previous_layer=self.layer_graph[-1])
        
        self.layer_graph.append(obj)

    def train(self,learning_rate, dataset, batch_size, num_epochs, loss_fn):
        num_iters = int(dataset.length/batch_size)
        if loss_fn != 'cross_entropy_with_softmax':
            loss_fn,loss_fn_derivative = loss_dict[loss_fn]
        else:
            loss_fn_derivative = loss_dict[loss_fn]
            loss_fn = None
        for _ in range(num_epochs):
            loss_log=[]
            for _ in range(num_iters):
                #Fwd pass
                X_train,y_train = dataset.next()
                out = self._forward_pass(X_train)
                
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

    def _forward_pass(self,data):
        # for layer in self.layer_graph[1:]:
        for layer in self.layer_graph:
            data = layer._forward_pass(data)
        return data

    def _backward_pass(self,loss_fn_derivative,out,y_train):
        # Gradient of final output wrt loss func
        grad = loss_fn_derivative(out,y_train)
        # for layer in reversed(self.layer_graph[1:]):
        for layer in reversed(self.layer_graph):
            grad = layer._backward_pass(grad)

    def _update_parameters(self,lr):
        for layer in self.layer_graph[1:]:
            if (type(layer).__name__ == 'Dense' or 
                    type(layer).__name__ == 'Conv'):
                layer.weights -= lr*layer.weights_grad
                layer.bias -= lr*layer.bias_grad
            elif type(layer).__name__ == 'BatchNorm':
                pass
        
    def predict(self):
        pass


        

