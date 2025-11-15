import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Node import Node

def main():
    input_tensor = Tensor(np.array([[1, 3]]), requires_gradient=False)
    
    # Can use nodes
    # L0_node0 = Node(Tensor(np.array([0.1, 0.2]), requires_gradient=True))
    # L0_node1 = Node(Tensor(np.array([0.3, 0.4]), requires_gradient=True))
    # L0_node2 = Node(Tensor(np.array([0.5, 0.6]), requires_gradient=True))
    # layer_0 = Layer(nodes=[L0_node0, L0_node1, L0_node2], bias=L0_bias, activation=Tensor.sigmoid)
    # layer_1 = Layer(nodes=[L1_node0], bias=L1_bias, activation=None)
    
    # Or can just use layers and skip using nodes altogether
    # (uses less Tensors)
    L0_weights = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])
    L0_weights = Tensor(L0_weights, requires_gradient=True)
    L0_bias = Tensor(np.array([1, 1, 1]), requires_gradient=True)
    L0 = Layer(weights=L0_weights, bias=L0_bias, activation=Tensor.sigmoid)
    
    L1_weights = np.array([
        [0.7, 0.8, 0.9]
    ])
    L1_weights = Tensor(L1_weights, requires_gradient=True)
    L1_bias = Tensor(np.array([1]), requires_gradient=True)
    L1 = Layer(weights=L1_weights, bias=L1_bias, activation=None)
    
    
    network = NeuralNetwork([L0, L1])
    
    z = network.forward(input_tensor)
    print(f"Network Output: {z.data}")
    
    z.backward()
    
    print(f"Final Gradient w.r.t Input: {input_tensor.grad}")
    
    
if __name__=="__main__":
    main()