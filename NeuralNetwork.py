import numpy as np
from Tensor import Tensor
from Layer import Layer
from config import DEBUG

class NeuralNetwork:
    def __init__(self, layers: list[Layer] = None):
        self.layers = layers
    
    def forward(self, batch_input_tensor: Tensor):
        output_tensor = batch_input_tensor
        
        for layer in self.layers:
            output_tensor = output_tensor @ layer.weights_tensor
            output_tensor = output_tensor + layer.bias_tensor
            if layer.activation:
                output_tensor = layer.activation(output_tensor)
            if DEBUG:
                print(f"Layer {layer._id} output_tensor:")
                print(output_tensor)
        
        return output_tensor
    
        