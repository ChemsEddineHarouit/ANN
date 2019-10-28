#MLP with 3 layers 1 In 1 Hidden and 1 Out

import pandas as pd
import numpy as np

from Utils import (sigmoid as f, tanh as t) 

class MLP_3:
    
    #Total of  neurons
    neurons_Hid1 = 8
    neurons_Hid2 = 8

    #Max Iterations
    MAX_ITER = 5000

    #Learning rate
    learning_rate = 0.3

    def __init__(self, dataFrame):
        self.X = dataFrame[0].T
        self.D = dataFrame[1].T
        #self.X = dataFrame.loc[:, dataFrame.columns != 'CLASS'].values.T
        #self.D = dataFrame.loc[:, dataFrame.columns == 'CLASS'].values.T
        self.N = self.X.shape[1]
        self.neurons_In  = np.shape(self.X)[0]
        self.neurons_Out = 1
        
        #test bias in W
        self.X = self.add_ones(self.X)
        
            

    def initParams(self):
        neurons_In, neurons_Hid1, neurons_Hid2, neurons_Out = (self.neurons_In, self.neurons_Hid1, self.neurons_Hid2, self.neurons_Out)
        W1 = np.random.randn(neurons_Hid1, neurons_In)
        b1 = np.zeros((neurons_Hid1, 1))
        W2 = np.random.randn(neurons_Hid2, neurons_Hid1)
        b2 = np.zeros((neurons_Hid2, 1))
        W3 = np.random.randn(neurons_Out, neurons_Hid2)
        b3 = np.zeros((neurons_Out, 1))

        #test bias in W
        print(np.shape(b1), np.shape(W1))
        W1 = np.concatenate((b1, W1), axis=1)

        self.params = {
            'W1'    : W1,
            'b1'    : b1,
            'W2'    : W2,
            'b2'    : b2,
            'W3'    : W3,
            'b3'    : b3
        } 
    
    def getAllParams(self):
        return [
            self.params.get('W1'),
            self.params.get('b1'),
            self.params.get('W2'),
            self.params.get('b2'),
            self.params.get('W3'),
            self.params.get('b3')
        ]
    
    def setAllParams(self, W1, b1, W2, b2, W3, b3):
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

    def forward_propagation(self):
        W1, b1, W2, b2, W3, b3 = self.getAllParams()
        #self.C1 = f(np.dot(W1, self.X) + b1)
        #self.C2 = f(np.dot(W2, self.C1) + b2)
        #self.C3 = f(np.dot(W3, self.C2) + b3)

        #test bias in W
        self.C1 = f(np.dot(W1, self.X))
        self.C2 = f(np.dot(W2, self.C1))
        self.C3 = f(np.dot(W3, self.C2))
        self.Y  = self.C3

    def calc_error(self):
        A = self.Y.dot((1- self.Y).T)
        B = (self.D - self.Y)
        C = A.dot(B)
        return C

    def backward_propagation(self):
        N = self.N
        W1, b1, W2, b2, W3, b3 = self.getAllParams()

        #Calculate Params Adjustments
        dZ3 = self.calc_error()
        dW3 = np.dot(dZ3, self.C2.T)
        db3 = np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = np.multiply(np.dot(W3.T, dZ3), self.C2 - np.power(self.C2, 2))
        dW2 = np.dot(dZ2, self.C1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), self.C1 - np.power(self.C1, 2))
        dW1 = np.dot(dZ1, self.X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        #Update Params
        r = self.learning_rate
        W1 = W1 - r*dW1
        b1 = b1 - r*db1
        W2 = W2 - r*dW2
        b2 = b2 - r*db2
        W3 = W3 - r*dW3
        b3 = b3 - r*db3
        self.setAllParams(W1, b1, W2, b2, W3, b3)

    def train(self):
        self.initParams()
        for i in range(self.MAX_ITER):
            self.forward_propagation()
            self.backward_propagation()
            if i % 100 == 0:
                print('Itération num %d'% i)

    def predict(self, dataFrame):
        #self.X = dataFrame.loc[:, dataFrame.columns != 'CLASS'].values.T
        #self.D = dataFrame.loc[:, dataFrame.columns == 'CLASS'].values.T
        self.X = dataFrame[0].T
        self.D = dataFrame[1].T
        self.X = self.add_ones(self.X)
        self.forward_propagation()

        print('Predicted   Desired  Result')
        Y = np.round(np.squeeze(self.Y))
        D = np.squeeze(self.D)
        nbTrue = 0
        for i in range(len(D)):
            result = Y[i] == D[i]
            if(result):
                nbTrue += 1 
            print('%d ... %d ... %d'%(Y[i], D[i], result))
        print('Accuracy = %d / 100'% (100*(nbTrue / len(D))))

    def add_ones(self, X):
        b = np.ones((np.shape(X)[1],1))
        X = np.concatenate((b.T, X), axis=0)
        return X

