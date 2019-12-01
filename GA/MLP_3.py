#MLP with 3 layers 1 In 1 Hidden and 1 Out

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from TimedProfile import calculate_time
from Utils import (sigmoid as f, tanh as t) 
from GA import GA
class MLP_3:
    
    #Total of  neurons
    neurons_Hid1 = 10

    #Max Iterations
    MAX_ITER = 10000

    #Learning rate
    learning_rate = 0.01


    def __init__(self, dataFrame):
        np.random.seed()
        self.X = dataFrame[0].T
        self.D = dataFrame[1].T
        self.N = self.X.shape[1]
        self.neurons_In  = np.shape(self.X)[0]
        self.neurons_Out = 1
            

    def getRandomConfig(self):
        neurons_In, neurons_Hid1, neurons_Out = (self.neurons_In, self.neurons_Hid1,  self.neurons_Out)
        W1 = np.random.randn(neurons_Hid1, neurons_In) *0.01
        b1 = np.zeros((neurons_Hid1, 1))
        W2 = np.random.randn(neurons_Out, neurons_Hid1) *0.01
        b2 = np.zeros((neurons_Out, 1))
        return (np.c_[b1, W1], np.c_[b2, W2])

    def initParams(self):
        
        W1, W2 = self.getRandomConfig()
     
        self.params = {
            'W1'    : W1,
            'W2'    : W2
        } 

    
    def getAllParams(self):
        return [
            self.params.get('W1'),
            self.params.get('W2')
        ]
    
    def setAllParams(self, W1, W2):
        self.params['W1'] = W1
        self.params['W2'] = W2

    def forward_propagation(self):
        W1, W2 = self.getAllParams()
        self.C1 = f(np.dot(W1, add_ones(self.X)))
        self.C2 = f(np.dot(W2, add_ones(self.C1)))
        self.Y  = np.squeeze(self.C2)

    @calculate_time
    def train(self):
        self.initParams()
        ga = GA(self)
        population = []
        for i in range(self.MAX_ITER):
            population = ga.selection(ga.population)
            parents = ga.select_parents(population)
            children = ga.crossover(parents)
            newPopulation = ga.mutation(children)
            population = newPopulation
        best_chromosome, best_fit = ga.get_best_chromosome(population)
        W1, W2 = ga.chromosome_to_matrix(best_chromosome)
        self.setAllParams(W1, W2)
        print(best_fit)
        
    def calc_error(self):
        return 0.5*np.sum((self.D - self.Y)**2)

    def predict(self, dataFrame):
        self.X = dataFrame[0].T
        self.D = dataFrame[1].T
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
        return 100*(nbTrue / len(D))

    def plotData(self):
        plt.scatter(self.X[0, :], self.X[1, :], c=self.D[:], s=40, cmap=plt.cm.Spectral);
        plt.title("IRIS DATA")
        plt.xlabel('Setal Length')
        plt.ylabel('Setal Width')
        plt.show()


def add_ones(matrix):
        ones = np.ones((1, np.shape(matrix)[1]))
        return np.r_[ones, matrix]
      
