from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
np.random.seed(2)

#Read Data
dataFrame = pd.read_csv('iris.data')

#Train and Test set
mask = np.random.rand(len(dataFrame)) < 0.8
train = dataFrame[mask]
test = dataFrame[~mask]

#Create the Neural Network and train it
X = train.loc[:, dataFrame.columns != 'CLASS'].values
D = train.loc[:, dataFrame.columns == 'CLASS'].values
D = np.squeeze(D)

# create mutli-layer perceptron classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(8,8), random_state=1)

# train
clf.fit(X, D)

# make predictions
X = test.loc[:, dataFrame.columns != 'CLASS'].values
D = test.loc[:, dataFrame.columns == 'CLASS'].values
D = np.squeeze(D)
Y = clf.predict(X)
Y = np.round(np.squeeze(Y), decimals=0)
D = np.squeeze(D)

nbTrue = 0
for i in range(len(D)):
    result = Y[i] == D[i]
    if(result):
        nbTrue += 1 
    print('%2d ... %2d ... %d'%(Y[i], D[i], result))
print('Accuracy = %d / 100'% (100*(nbTrue / len(D))))
