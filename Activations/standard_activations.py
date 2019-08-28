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
    return np.nan_to_num(A)

def sigmoid_backward(gradients, Z):
	temp = np.nan_to_num(np.exp(-Z))
	temp = np.nan_to_num(np.divide(temp,(1+temp)**2))
	Z_grad = np.multiply(gradients,temp)
	return Z_grad

# def softmax(Z):
#     A = np.exp(Z)/np.sum(np.exp(Z))
#     return A,Z

def relu(Z):
	A = np.where(Z>0, Z, 0)
	return A
	
def relu_backward(gradients, Z):
    Z_grad = np.where(Z>0, gradients, 0)
    return Z_grad

def leaky_relu(Z,alpha=0.1):
    A = np.where(Z>0,Z,alpha*Z)
    return A

def leaky_relu_backward(gradients, Z, alpha=0.1):
    Z_grad = np.where(Z>0,gradients,alpha*gradients)
    return Z_grad

def linear(Z):
    return Z

def linear_backward(gradients,Z):
    return gradients