import numpy as np

def sigmoid(Z):
	# For any value z <= -710 this returns NaN
    # A = 1/(1+np.exp(-Z))
    
    # More stable
    A = np.where(
        Z>0, 
        1/(1+np.exp(-Z)), 
        np.exp(Z)/(1+np.exp(Z))
    )
    return A

def softmax(Z):
    A = np.exp(Z)/np.sum(np.exp(Z))
    return A,Z

def relu(Z):
	A = np.where(Z>0, Z, 0)
	return A

def sigmoidBackward(dA, activation_cache):
	temp = np.exp(-activation_cache)
	temp = np.nan_to_num(np.divide(temp,(1+temp)**2))
	dZ = np.multiply(dA,temp)
	return dZ
	
def reluBackward(dA, activation_cache):
    temp = 1.0*(activation_cache>0)
    dZ = np.multiply(dA,temp)
    return dZ