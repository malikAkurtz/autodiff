import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Node import Node

def main():
    input_tensor = Tensor(np.array([[1, 2]]), requires_gradient=False)
    
    node_0 = Node(Tensor(np.array([0.1, 0.2]), requires_gradient=True))
    node_1 = Node(Tensor(np.array([0.3, 0.4]), requires_gradient=True))
    node_2 = Node(Tensor(np.array([0.5, 0.6]), requires_gradient=True))
    
    layer_0 = Layer(nodes=[node_0, node_1, node_2], bias=Tensor(np.array([1, 1, 1]), requires_gradient=True))
    
    network = NeuralNetwork([layer_0])
    
    f = network.forward(input_tensor)
    
    f.backward()
    
    print(input_tensor.grad)
    
    
if __name__=="__main__":
    main()