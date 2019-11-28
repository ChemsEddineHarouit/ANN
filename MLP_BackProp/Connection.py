import numpy as np



# a connection is the link between the actual neuron and another neuron from the previous layer
# so the connection will store the previous neuron, the weight and the dWeight which is the error of the weight
class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0
