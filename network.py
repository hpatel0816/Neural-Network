import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#Sample input data
# X = [[1.0, 2.0, 3.0, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100,3)

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        #The 0.1 is used to normalize the inputs between -1 & 1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        #Biases initiazed as 0
        self.baises = np.zeros((1, n_neurons))

    def foward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises


class ReLU_Activation:
    def forward_pass(self, inputs):
        self.output = np.maximum(0, inputs)

#Initialize the neural network layers & activation func
layer1 = Dense_Layer(2,5)
#layer2 = Dense_Layer(5,2)
activation1 = ReLU_Activation()


#Forward pass inputs through network layers & activation func
layer1.foward_pass(X)
print(layer1.output)
#layer2.foward_pass(layer1.output)
#print(layer2.output)
activation1.forward_pass(layer1.output)
print(activation1.output)