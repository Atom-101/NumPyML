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