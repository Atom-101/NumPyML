import numpy as np


def mean_squared_error(y_predicted,y):
    squared_error = np.square(y_predicted-y)
    mse = np.mean(squared_error)
    return mse

def mean_squared_error_derivative(y_predicted,y):
    return 2*(y_predicted-y)

def binary_cross_entropy(y_predicted,y):
    return np.mean(
        np.multiply(y,-np.log(y_predicted)) +
        np.multiply(1-y,-np.log(1-y_predicted))
    )

def binary_cross_entropy_derivative(y_predicted,y):
    m = y.shape[0]
    numerator = y_predicted-y
    denominator = m*(y_predicted-np.square(y_predicted)) + 1e-7
    return np.divide(
        numerator,
        denominator
    )