import numpy as np

class MeanSquaredError():
    @classmethod
    def loss(cls,y_predicted,y):
        squared_error = np.square(y_predicted-y)
        mse = np.mean(squared_error)
        return mse
    @classmethod
    def loss_derivative(cls,y_predicted,y):
        return 2*(y_predicted-y)

class BinaryCrossEntropy():
    @classmethod
    def loss(cls,y_predicted,y):
        return np.mean(
            np.multiply(y,-np.log(y_predicted)) +
            np.multiply(1-y,-np.log(1-y_predicted))
        )
    @classmethod
    def loss_derivative(cls,y_predicted,y):
        m = y.shape[0]
        numerator = y_predicted-y
        denominator = m*(y_predicted-np.square(y_predicted)) + 1e-7
        return np.divide(
            numerator,
            denominator
        )
        
def softmax_cross_entropy_with_logits(logits,y):
    shifted_logits = logits - np.max(logits,axis=1,keepdims=True)
    Z = np.sum(np.exp(shifted_logits),axis=1,keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = logits.shape[0]
    loss = -np.sum(log_probs[np.arange(N),y.argmax(axis=-1)])/N
    gradients = probs
    gradients[np.arange(N),y.argmax(axis=-1)] -= 1
    gradients/=N
    return loss,gradients
