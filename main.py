import numpy as np
from sklearn import datasets
from MLP import MLP as MLP

#Read Data
iris = datasets.load_iris()

X = iris.data
D = iris.target


#Train and Test set
np.random.seed(1)
mask = np.random.rand(len(X)) < 0.9
trainData = trainX, trainD = X[mask], D[mask]
testData = testX , testD  = X[~mask], D[~mask]

#Create the Neural Network and train it
mlp = MLP(trainX, trainD, [2, 5], 0.1)
mlp.train()

#Predict with the test Data
Acc = mlp.predict(trainX, trainD)
print("Accuracy = %d"%(Acc))
mlp.plotErrors()
print("Last error = %d"%(mlp.costs[-1]))


# neurons = []
# rates = [.001, .005, .01, .04, .08, .1, .4, .8, 1.2]
# Acc = acc = 0
# for i in range(1, 20):
#     for j in range(1, 20):
#         for k in range(1, 2):
#             for r in rates:
#                 mlp = MLP(trainX, trainD, [i, j], r)
#                 mlp.train()
#                 acc = mlp.predict(testX , testD)
#                 if(acc == Acc):
#                     neurons.append((i,j,k,r))
#                 elif (acc > Acc):
#                     Acc = acc
#                     neurons = []
#                     neurons.append((i,j,k,r))
#                 print(  "\n\n\n\n", i, j, k, r)
#                 print("Best accuracy is %d/100 with neurons:\nL1, L2, L3, rate\n%s"%(Acc, neurons[-1]))
# print("Best accuracy is %d/100 with neurons:\nL1, L2, L3, rate\n%s"%(Acc, neurons))
