import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Node import Node

def main():
    input_tensor = Tensor(np.array([[1, 3]]), requires_gradient=False)
    
    L0_node0 = Node(Tensor(np.array([0.1, 0.2]), requires_gradient=True))
    L0_node1 = Node(Tensor(np.array([0.3, 0.4]), requires_gradient=True))
    L0_node2 = Node(Tensor(np.array([0.5, 0.6]), requires_gradient=True))
    L0_bias = Tensor(np.array([1, 1, 1]), requires_gradient=True)
    
    L1_node0 = Node(Tensor(np.array([0.7, 0.8, 0.9]), requires_gradient=True))
    L1_bias = Tensor(np.array([1]), requires_gradient=True)
    
    
    
    layer_0 = Layer(nodes=[L0_node0, L0_node1, L0_node2], bias=L0_bias)
    layer_1 = Layer(nodes=[L1_node0], bias=L1_bias)
    
    network = NeuralNetwork([layer_0, layer_1])
    
    f = network.forward(input_tensor)
    print(f"Network Output: {f.data}")
    
    f.backward()
    
    
    print(f"Final Gradient: {input_tensor.grad}")
    
    
if __name__=="__main__":
    main()