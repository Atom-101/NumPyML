import numpy as np


def mean_squared_error(y_predicted,y):
    squared_error = np.square(y_predicted-y)
    mse = np.mean(squared_error)
    return mse

def mean_squared_error_derivative(y_predicted,y):
    return 2*(y_predicted-y)
