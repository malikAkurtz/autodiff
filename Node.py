from Tensor import Tensor
from debug import DEBUG

class Node:
    global_id = 0
    def __init__(self, weights: Tensor):
        self.weights = weights
        self._id = Node.global_id
        Node.global_id += 1
        