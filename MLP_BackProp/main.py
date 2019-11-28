import numpy as np
from sklearn import datasets

#Read Data
iris = datasets.load_iris()

X = iris.data
D = iris.target


#Train and Test set
np.random.seed(2)
mask = np.random.rand(len(X)) < 0.9
trainData = trainX, trainD = X[mask], D[mask]
testData = testX , testD  = X[~mask], D[~mask]
trainD = trainD.reshape(len(trainD), 1)
testD  = testD.reshape(len(testD), 1)


#Creating Network
from Network import Network
from Neuron import Neuron
topology = []
topology.append(4)
topology.append(4)
topology.append(1)
net = Network(topology)
Neuron.r = 2
err = 100
while err > 47:

    err = 0
    inputs = trainX
    outputs = trainD
    for i in range(len(inputs)):
        net.setInput(inputs[i])
        net.feedForword()
        net.backPropagate(outputs[i])
        err = err + net.getError(outputs[i])
    print ("error: ", err)


inputs = testX
outputs = testD
Y = []
for i in range(len(inputs)):
    net.setInput(inputs[i])
    net.feedForword()
    Y.append(net.getY())

Y = np.round(np.squeeze(Y))
D = np.squeeze(outputs)
nbTrue = 0
for i in range(len(D)):
    result = Y[i] == D[i]
    if(result):
        nbTrue += 1 
    print('%d ... %d ... %d'%(Y[i], D[i], result))
print('Accuracy = %d / 100'% (100*(nbTrue / len(D))))