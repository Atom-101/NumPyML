import numpy as np

class BatchNorm(object):
    "Instantiate a batch normalization layer"
    def __init__(self,momentum):
        self.momentum = momentum

    def _init_weights(self,previous_layer):
        if previous_layer.weights.ndim == 2:
            dim = previous_layer.weights.shape[1]
            self.units = previous_layer.units
        else:
            dim= previous_layer.weights.shape[0]
            self.output_height,self.output_width,self.filters = previous_layer.output_height,previous_layer.output_width,previous_layer.filters
        self.running_mean = np.zeros(dim)    
        self.running_var = np.zeros(dim)
        self.gamma = np.ones(dim)    
        self.beta = np.zeros(dim)

    def _forward_pass(self,X,train):
        if train:
            mean = np.mean(X.reshape(-1,X.shape[-1]),axis=0)
            if X.ndim == 4:
                X_zero_mean = X - mean[np.newaxis,np.newaxis,np.newaxis,:]
            elif X.ndim == 2:
                X_zero_mean = X - mean[np.newaxis,:]
            
            var = np.var(X.reshape(-1,X.shape[-1]),axis=0)
            stddev = np.sqrt(var+1e-7)
            
            if X.ndim == 4:
                X = np.divide(X_zero_mean,stddev[np.newaxis,np.newaxis,np.newaxis,:])
            elif X.ndim == 2:
                X = np.divide(X_zero_mean,stddev[np.newaxis,:])
            
            self.X_normalized = X
            self.var = var
            self.stddev = stddev
            
            if X.ndim == 4:
                self.X_zero_mean = lambda: self.X_normalized * self.stddev[np.newaxis,np.newaxis,np.newaxis,:]
            elif X.ndim == 2:
                self.X_zero_mean = lambda: self.X_normalized * self.stddev[np.newaxis,:]
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            if X.ndim == 4:    
                X = np.divide(
                    X - self.running_mean[np.newaxis,np.newaxis,np.newaxis,:],
                    np.sqrt(self.running_var[np.newaxis,np.newaxis,np.newaxis,:])
                )
            elif X.ndim == 2:
                X = np.divide(
                    X - self.running_mean[np.newaxis,:],
                    np.sqrt(self.running_var[np.newaxis,:])
                )
        
        if X.ndim == 4:
            X = self.gamma[np.newaxis,np.newaxis,np.newaxis,:] * X + self.beta[np.newaxis,np.newaxis,np.newaxis,:]
        elif X.ndim == 2:
            X = self.gamma[np.newaxis,:] * X + self.beta[np.newaxis,:]
        return X

    def _backward_pass(self,gradients):
        self.beta_grad = np.sum(gradients.reshape(-1,gradients.shape[-1]),axis=0)
        self.gamma_grad = np.sum((gradients * self.X_normalized).reshape(-1,gradients.shape[-1]),axis=0)
        
        if gradients.ndim == 4:
            gradients = self.gamma[np.newaxis,np.newaxis,np.newaxis,:] * gradients
        elif gradients.ndim == 2:
            gradients = self.gamma[np.newaxis,:] * gradients
        
        inverse_stddev_grad = np.sum((self.X_zero_mean() * gradients).reshape(-1,gradients.shape[-1]), axis=0)
        sttdev_grad = -1./np.square(self.stddev) * inverse_stddev_grad
        var_grad = (0.5) * 1/np.sqrt(self.var+1e-7) * sttdev_grad
        
        if self.X_normalized.ndim == 4:    
            X_zero_mean_squared_grad = np.ones_like(self.X_normalized) * var_grad[np.newaxis,np.newaxis,np.newaxis,:]
        elif self.X_normalized.ndim == 2:
            X_zero_mean_squared_grad = np.ones_like(self.X_normalized) * var_grad[np.newaxis,:]
        
        X_zero_mean_grad_1 = 2 * self.X_zero_mean() * X_zero_mean_squared_grad

        if self.X_normalized.ndim == 4:
            X_zero_mean_grad_2 = np.divide(gradients,self.stddev[np.newaxis,np.newaxis,np.newaxis,:])
        elif self.X_normalized.ndim == 2:
            X_zero_mean_grad_2 = np.divide(gradients,self.stddev[np.newaxis,:])

        X_zero_mean_grad = X_zero_mean_grad_1 + X_zero_mean_grad_2 # = X_grad_1
        mean_grad = np.sum(X_zero_mean_grad.reshape(-1,gradients.shape[-1]), axis=0)

        X_grad_2 = (1./self.X_normalized.shape[-1]) * np.ones_like(self.X_normalized) * mean_grad
        X_grad = X_grad_2 + X_zero_mean_grad

        return X_grad