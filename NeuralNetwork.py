import numpy as np
from Tensor import Tensor
from Layer import Layer
from Node import Node
from debug import DEBUG

class NeuralNetwork:
    def __init__(self, layers: list[Layer] = None):
        self.layers = layers
    
    def forward(self, batch_input: Tensor):
        output = batch_input
        
        for layer in self.layers:
            if DEBUG:
                print(f"Current output state:")
                print(output.data)
                print(f"Has shape: {output.data.shape}")
                print(f"Layer {layer._id} weights matrix (Tensor {layer.weights_matrix._id}):")
                print([layer.weights_matrix.data])
                print(f"Has shape: {layer.weights_matrix.data.shape}")
                print(f"Layer {layer._id} bias vector (Tensor {layer.bias._id}):")
                print(layer.bias.data)
                print(f"Has shape: {layer.bias.data.shape}")
            output @= Tensor.transpose(layer.weights_matrix)
            output += layer.bias
        
        return output
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)