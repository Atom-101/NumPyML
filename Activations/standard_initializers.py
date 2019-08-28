import numpy as np
import math

def gaussian_init(size,**kwargs):
    return np.random.normal(
        loc=0,
        scale=0.1,
        size=size
    )

def kaiming_normal_init(size,**kwargs):
    return np.random.normal(
        loc=0,
        scale=math.sqrt(2)/math.sqrt(kwargs['n']),
        size=size
    )

def kaiming_uniform_init():
    pass