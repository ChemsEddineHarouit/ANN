import numpy as np
from sklearn import datasets
from MLP  import MLP as NeuralNetwork
#Read Data
dataFrame = datasets.load_iris()
# For digits dataset uncomment the following line
# dataFrame = datasets.load_digits()

X = dataFrame.get('data')
D = dataFrame.get('target')




#Train and Test set
np.random.seed(2)
mask = np.random.rand(len(X)) < 0.8
trainData = trainX, trainD = X[mask], D[mask]
testData = testX , testD  = X[~mask], D[~mask]


# #################### XOR
# trainX = np.array([[0,0], [0,1], [1,0], [1,1]])
# trainD = np.array([0, 1, 1, 0])


#Create the Neural Network and train it
nn = NeuralNetwork([4, 7, 5, 1], 0.1)
nn.errors = []
nn.accurs = []

nn.train(X, D, 1000)

#Predict
Y = nn.run(testX, testD)

print('Accuracy : %d, Error : %d'% (nn.accus[-1], nn.errors[-1]))

nn.plotAcc()
nn.plotErrors()
