import neuralnetwork as nn
import numpy as np

layer_sizes = (3,5,10)
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)