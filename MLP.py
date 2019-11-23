import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
from TimedProfile import calculate_time
from Utils import *

"""
the MultiLayerPerceptron has multiple parameters:
MAX_ITER        : is the number of training
                    the training is feedforward followed by backprop functions
learnin_rate    : is the rate changing of the weights
costs           : is a list of errors in cevery iteration and it's length is obviously MAX_ITER


to create a MLP we need :   X (data) D(desired output) neurons_Hi(list of numbers of neurons of hidden layers)
                            neurons_in (number of neurons of the input layer = number of columns of X)
                            neurons_ou (number of neurons of the output layer= number of rows of Y or D)
                            Y is the output calculated by the MLP (if Y = D then we have a 100% accuracy)
                            W1, W2, W3 are the weight matrices
                            b1, b2, b3 are the biases (we integrate b in the first column of W)
                            W are initialized in random numbers while the b are initialized with 0
                            C stands for Couche and that represents the layers of the MLP
the feedforward is calculated as : C(i) = sigmoid(W(i)*C(i-1) + b(i)) and C(0) = X the input data
                            but while the b is integrated in W we just need to calculate C(i) = sigmoid(W(i)*C(i-1))
                            but we have to add a row in C(i-1) before each operation
the backward prop is calculated as:
                            dZ(i) refers to the error of the layer "i" / C(i) = sigmoid(Z(i)) but we don't use this formula
                                dZ(i) = W(i).T * dZ(i+1) * sigmoid_derivative(C(i)) with dZ(last layer) = D - Y
                            dW(i) refers to the error done in the weights W(i)
                                dWi(i) = dZ(i) * C(i-1).T
                            we have to adjust weights
                                W(i) += learning_rate * dW(i)

Matrices Shapes:
    I= neurons_in ; H= neurons_hi ; O= neurons_ou; N: number of samples

    X (I, N)  D (1, N)
    W1(H, I)  W2(O, H)
    C1(H, N)  C2(O, N)
    when we add a column to a matrix we add _c in the name
    when we add a row to a matrix we add _r in the name
        NB: remember that we add rows of 1 in C(i) and X while we add columns of b permanently in the W(i)
    so:
    feedforward:
        C(i) = sigmoid(W_c(i)*C_r(i-1)) // you can verify the shapes
    backprop:
        dWi(i) = dZ(i) * C_r(i-1).T
        dZ(i) = W(i).T * dZ(i+1) * sigmoid_derivative(C(i)) // notice that we use here (W and C) not (W_c and C_r) so we have to remove the 1 of C and the bias from W
        W_c(i) += learning_rate * dW(i)

"""

class MLP:
    MAX_ITER    = 10000
    learning_rate = 0.0000001
    costs = []

    def __init__(self, X, D, neurons_Hi, r):
        self.learning_rate = r
        self.X           = X.T
        neurons_in       = X.shape[1]
        neurons_ou       = 1

        self.neurons_hi1 = neurons_Hi[0]
        self.neurons_hi2 = neurons_Hi[1]

        W1               = np.random.rand(self.neurons_hi1,        neurons_in) *0.1
        b1               = np.zeros((self.neurons_hi1, 1))
        W2               = np.random.rand(self.neurons_hi2     ,   self.neurons_hi1) *0.1      
        b2               = np.zeros((self.neurons_hi2, 1))
        W3               = np.random.rand(neurons_ou     ,   self.neurons_hi2) *0.1      
        b3               = np.zeros((neurons_ou, 1))
        self.W1          = np.c_[b1, W1]
        self.W2          = np.c_[b2, W2]
        self.W3          = np.c_[b3, W3]
        self.D           = D


        self.Y          = np.zeros(self.D.shape)
        print(' ANN : (%d, %d, %d, %d)'%(neurons_in, self.neurons_hi1, self.neurons_hi2, neurons_ou))
        print(' Weights : W1:%s and W2:%s'%(self.W1.shape, self.W2.shape))
        print(' Data : X:%s and D:%s'%(X.shape, D.shape))

    def feedforward(self):
        self.C1 = sigmoid(np.dot(self.W1, add_1_r(self.X)))
        self.C2 = sigmoid(np.dot(self.W2, add_1_r(self.C1)))
        self.C3 = sigmoid(np.dot(self.W3, add_1_r(self.C2)))
        self.Y = self.C3

    def calc_cost(self, err):
        cost = 0.5 * np.sum(err**2)
        self.costs.append(cost)

    def calc_error(self):
        err = ((self.D - self.Y))
        self.calc_cost(err)
        return err

    def backprop(self):

        dZ3 = self.calc_error()
        dW3 = np.dot(dZ3, add_1_r(self.C2).T)

        dZ2 = np.multiply(np.dot(rem_1st_c(self.W3).T, dZ3), sigmoid_deriv(self.C2))
        dW2 = np.dot(dZ2, add_1_r(self.C1).T)

        dZ1 = np.multiply(np.dot(rem_1st_c(self.W2).T, dZ2), sigmoid_deriv(self.C1))
        dW1 = np.dot(dZ1, add_1_r(self.X).T)


        # update the weights with the derivative (slope) of the loss function
        self.W1 += self.learning_rate * dW1
        self.W2 += self.learning_rate * dW2
        self.W3 += self.learning_rate * dW3
    
    @calculate_time
    def train(self):
        for i in range(self.MAX_ITER):
            self.feedforward()
            self.backprop()
            # print(np.equal(before, self.W1))
    
    def predict(self, X, D):
        self.X = X.T
        self.D = D
        self.feedforward()

        # print('Predicted   Desired  Result')
        Y = np.round(np.squeeze(self.Y))
        D = np.squeeze(self.D)
        nbTrue = 0
        for i in range(len(D)):
            result = Y[i] == D[i]
            if(result):
                nbTrue += 1 
        #     print('%d ... %d ... %d'%(Y[i], D[i], result))
        # print('Accuracy = %d / 100'% (100*(nbTrue / len(D))))
        return 100*(nbTrue / len(D))

    def plotErrors(self):
        errors = self.costs
        plt.plot(errors);
        plt.title("IRIS Erros")
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.show()
