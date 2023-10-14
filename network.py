import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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
        #All inputs <= 0 are set to 0, otherwise they maintain their value
        self.output = np.maximum(0, inputs)


class Softmax_Activation:
    def forward_pass(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        #Compute the batch loss and determine overall neural network loss
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Categorical_Cross_Entropy_Loss(Loss):
    def forward(self, y_pred, y_actual):
        samples = len(y_pred)
        #Adjust the layer output values to avoid miscalculation (divide by 0 error)
        y_pred_adjusted = np.clip(y_pred, 1e-7, 1-1e-7)

        #Map samples from batch to actual values depending on size of y_act
        if len(y_actual.shape) == 1:
            #1D array of y_act
            confidences = y_pred_adjusted[range(samples), y_actual]
        elif len(y_actual.shape) == 2:
            #One-hot-encoded (2D) array of y_act
            confidences = np.sum(y_pred_adjusted * y_actual, axis=1)

        #Determine the likehood of each prediction using the negative log method
        probabilities = -np.log(confidences)
        return probabilities

#Create dataset of coordinate pairs
X, y = spiral_data(samples=100, classes=3)

#Initialize the neural network layers & activation funcs.
'''
2 inputs (x and y coordinates) are passed for first layer and it outputs 3 (arbitrarily chosen).
Hence, the second layer takes 3 inputs and outputs 3 corresponding to each of the 3 classes.
'''
layer1 = Dense_Layer(2,3)
activation1 = ReLU_Activation()

layer2 = Dense_Layer(3,3)
activation2 = Softmax_Activation()


#Feed the data through the neural network and create predictions
layer1.foward_pass(X)
activation1.forward_pass(layer1.output)

layer2.foward_pass(layer1.output)
activation2.forward_pass(layer2.output)

#print(activation2.output[:5])

#Compute Loss
loss_func = Categorical_Cross_Entropy_Loss()
loss = loss_func.calculate(activation2.output, y)

print("Loss:", loss)