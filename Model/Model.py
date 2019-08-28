import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import numpy as np
from Losses.Losses import *
from tqdm import trange

loss_dict = {
    'mse': (mean_squared_error,mean_squared_error_derivative),
    'binary_cross_entropy' : (binary_cross_entropy,binary_cross_entropy_derivative),
    'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits
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
        self.loss_log = []
        for epoch in range(num_epochs):
            epoch_progress = trange(num_iters,desc=f'Epoch [{epoch+1}/{num_epochs}]')
            epoch_log = []
            for _ in epoch_progress:
                #Fwd pass
                X_train,y_train = dataset.next()
                out = self._forward_pass(X_train,True)
                #Compute loss
                if loss_fn:
                    loss = loss_fn(out,y_train)
                    epoch_log.append(loss)
                    # @todo: write plot code

                #Backwd pass
                if loss_fn:
                    self._backward_pass(loss_fn_derivative,out,y_train)
                else:
                    loss = self._backward_pass(loss_fn_derivative,out,y_train)
                    epoch_log.append(loss)
                
                #Update params
                self._update_parameters(learning_rate)
                epoch_progress.set_description(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {loss: .5f} | Av Loss: {np.mean(np.array(epoch_log)):.5f}')
                epoch_progress.refresh()
            print()
            self.loss_log.append(epoch_log)

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
        loss = None
        try:    
            loss,grad = loss_fn_derivative(out,y_train)
        except:
            grad = loss_fn_derivative(out,y_train)
        # for layer in reversed(self.layer_graph[1:]):
        for layer in reversed(self.layer_graph):
            grad = layer._backward_pass(grad)
        return loss

    def _update_parameters(self,lr,weight_decay=1e-3):
        for layer in self.layer_graph:
            if (type(layer).__name__ == 'Dense' or 
                    type(layer).__name__ == 'Conv'):
                if weight_decay:
                    layer.weights *= (1-lr*weight_decay/32)
                layer.weights -= lr*layer.weights_grad
                layer.bias -= lr*layer.bias_grad
            elif type(layer).__name__ == 'BatchNorm':
                layer.gamma -= lr*layer.gamma_grad
                layer.beta -= lr*layer.beta_grad
        
    def predict(self):
        "Predict using trained model. To be implemented"
        raise NotImplementedError('Predict is not yet implemented')


        

