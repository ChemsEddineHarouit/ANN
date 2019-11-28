import numpy as np 

def tanh(x):
    return np.tanh(x)

def sigmoid_deriv(x):
    return x*(1-x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x * 1.0))
            
def add_1_r(matrix):
    l = np.shape(matrix)[1]
    ones = np.ones((1,l))
    new =  np.r_[ones, matrix]
    return new

def add_1_c(matrix):
    l = np.shape(matrix)[0]
    ones = np.ones((l,1))
    new =  np.c_[ones, matrix]
    return new

def rem_1st_c(matrix):
    new = matrix[:, 1:]
    return new

def rem_1st_r(matrix):
    new = matrix[1:, :]
    return new


def printShapes(s, A, dict = False):
    print(s, end='   ')
    if(dict):
        for k in A.keys():
            print('%d '%(k+1), A[k].shape, ' ', end='')
        print('')
    else:
        for a in A:
            print(a.shape, ' ', end='')
        print('')
