import numpy as np
from matplotlib import pyplot as plt


class MLP(object):

    accus  = []
    errors = []
    W      = {}

    def __init__(self, topology, learning_rate):
        
        for i in range(1, len(topology)):
            nodes_in  = topology[i-1]
            nodes_ou  = topology[i]
            self.W[i] = np.random.normal(0.0, nodes_in**-0.5, 
                                       (nodes_in, nodes_ou))
        self.last_layer = len(topology) - 1
        self.lr = learning_rate
        self.sigmoid        = lambda x : 1/(1+np.exp(-x))  
        self.sigmoid_deriv  = lambda x : x*(1-x)  
                    


    def train(self, X, D, ITERATIONS = 1000):
        self.accus = []
        self.errors = []
        for i in range(ITERATIONS):
            self.train_once(X, D)
            if(i%100 == 0):
                print('Iteration : %d'% i)

    def init_dW(self):
        dW = {}
        for i, W in self.W.items():
            dW[i] = np.zeros(W.shape)
        return dW

    def train_once(self, X, D):
        N  = X.shape[0]
        dW = self.init_dW()
        Y  = []
        for X, y in zip(X, D):
            C  = self.forward_pass_train(X)
            Y.append(C[self.last_layer])
            dW = self.backpropagation(C, X, y, dW)
        self.update_weights(dW, N)
        self.errors.append(self.calc_error(D, Y))
        self.accus.append(self.calc_acc(D, Y))


    def forward_pass_train(self, X):
        C = {}
        C[0] = X
        for i,W in self.W.items():
            Zi   = np.dot(C[i-1], W)
            if(i == self.last_layer):
                C[i] = Zi #activation function here is f(x) = x
            else:
                C[i] = self.sigmoid(Zi)

        return C

    def backpropagation(self, C, X, y, dW):
        dC = {}
        for i in range(self.last_layer, 0, -1):
            if(i == self.last_layer):
                error      = y - C[i]
                dC[i] = error #because df/dx = x (last layer the activation function f(x) = x)
            else:
                error      = np.dot(dC[i+1], self.W[i+1].T)
                dC[i] = error * self.sigmoid_deriv(C[i])
            dW[i]  = dW[i]  + dC[i] * C[i-1][:, None]
        return dW

    def update_weights(self, dW, N):
            for i,dW in dW.items():
                self.W[i] = self.W[i] + self.lr * dW / N

    def run(self, X, D):
        C = self.forward_pass_train(X)
        Y = C[self.last_layer]
        self.accus.append(self.calc_acc(D, Y, verbose=True))
        self.errors.append(self.calc_error(D, Y))
        return Y

    def calc_error(self, D, Y):
        Y = np.squeeze(Y)
        error = 0.5 * np.sum(np.square(D - Y))
        return error
    
    def calc_acc(self, D, Y, verbose = False,):
        if(verbose):
            print('Predicted   Desired  Result')
        Y = np.round(np.squeeze(Y))
        D = np.squeeze(D)
        nbTrue = 0
        for i in range(len(D)):
            result = Y[i] == D[i]
            if(result):
                nbTrue += 1 
            if(verbose):
                print('%d ... %d ... %d'%(Y[i], D[i], result))
        if(verbose):
            print('Accuracy = %d / 100'% (100*(nbTrue / len(D))))
        return 100*(nbTrue / len(D))

  
    def plotErrors(self):
        plt.plot(self.errors)
        plt.title("IRIS Erros")
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.show()
        
    def plotAcc(self):
        plt.plot(self.accus)
        plt.title("IRIS Accuracy")
        plt.xlabel('iteration')
        plt.ylabel('Accuracy')
        plt.show()