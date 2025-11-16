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
                print(f"Layer {layer._id} weights matrix (Tensor {layer.weights._id}):")
                print([layer.weights.data])
                print(f"Has shape: {layer.weights.data.shape}")
                print(f"Layer {layer._id} bias vector (Tensor {layer.bias._id}):")
                print(layer.bias.data)
                print(f"Has shape: {layer.bias.data.shape}")
                print(f"Layer activation: {layer.activation}")
            output = output @ layer.weights
            output = output + layer.bias
            if layer.activation:
                output = layer.activation(output)
        
        return output
    
        