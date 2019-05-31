import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def backward_sigmoid(a):
    return a * (1 - a)
