import numpy as np
from sklearn import datasets
from MLP import MLP as MLP
from MLP_2Layers import MLP as MLP2

#Read Data
dataFrame = datasets.load_iris()
# For digits dataset uncomment the following line
# dataFrame = datasets.load_digits()

X = dataFrame.get('data')
D = dataFrame.get('target')




#Train and Test set
np.random.seed(1)
mask = np.random.rand(len(X)) < 0.8
trainData = trainX, trainD = X[mask], D[mask]
testData = testX , testD  = X[~mask], D[~mask]


# #################### XOR
# trainX = np.array([[0,0], [0,1], [1,0], [1,1]])
# trainD = np.array([0, 1, 1, 0])


#Create the Neural Network and train it
mlp2 = MLP2(trainX, trainD)
mlp2.train()


#Predict with the test Data
Acc = mlp2.predict(testX, testD)
print("Accuracy = %d"%(Acc))
print("Last errors = %s"%(mlp2.costs[-5:]))
print("Smallest error = %2.2f"%(min(mlp2.costs)))
mlp2.plotErrors()
mlp2.plotAcc()


mlp = MLP(trainX, trainD, [3, 5], np.exp(-10), 5000)
mlp.train()
Acc = mlp.predict(testX, testD)
print("Accuracy = %d"%(Acc))
print("Last errors = %s"%(mlp.costs[-5:]))
print("Smallest error = %2.2f"%(min(mlp.costs)))
mlp.plotErrors()
mlp.plotAcc()