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
np.random.seed()
mask = np.random.rand(len(X)) < 0.8
trainData = trainX, trainD = X[mask], D[mask]
testData = testX , testD  = X[~mask], D[~mask]
inputShape = np.shape(X)[1]
if(np.ndim(D) > 1):
    outputShape = np.shape(D)[1]
else:
    outputShape = 1

# #################### XOR
# trainX = np.array([[0,0], [0,1], [1,0], [1,1]])
# trainD = np.array([0, 1, 1, 0])


#Create the Neural Network and train it
nn = NeuralNetwork([inputShape, 2, 5, outputShape], 0.1)
nn.train(X, D, 1000)

#Predict
Y = nn.run(testX, testD)

print('Accuracy : %d %%, Error : %f'% (nn.accus[-1], nn.errors[-1]))

nn.plotAcc()
nn.plotErrors()
