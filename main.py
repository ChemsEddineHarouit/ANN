import pandas as pd
import numpy as np
from sklearn import datasets
from MLP_3 import MLP_3

np.random.seed(2)

#Read Data
iris = datasets.load_iris()
X = iris.data
D = iris.target



# dataFrame = pd.read_csv('iris.data')

#Train and Test set
mask = np.random.rand(len(X)) < 0.8
train = X[mask], D[mask]
test = X[~mask], D[~mask]

#Create the Neural Network and train it
mlp_3 = MLP_3(train)
mlp_3.plotData()
mlp_3.train()

#Predict with the test Data
mlp_3.predict(test)
