import numpy as np

class Flatten(object):
    def __init__(self, arr):
        self.arr = arr
        self.units = np.prod(arr.shape[1:])
    
    def _forward_pass(self):
        shape = self.arr.shape
        return np.reshape(self.arr,(shape[0],-1))

    def _backward_pass(self):
        return None


