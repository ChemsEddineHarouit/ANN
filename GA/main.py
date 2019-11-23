import pandas as pd
import numpy as np
from sklearn import datasets
from MLP_3 import MLP_3
from sklearn.feature_selection import VarianceThreshold

#Read Data
iris = datasets.load_iris()

X = iris.data
D = iris.target


#Train and Test set
np.random.seed()
mask = np.random.rand(len(X)) < 0.8
trainData = X[mask], D[mask]
testData = X[~mask], D[~mask]

#Create the Neural Network and train it
mlp_3 = MLP_3(trainData)
# mlp_3.plotData()
mlp_3.train()

#Predict with the test Data
mlp_3.predict(trainData)

