from __future__ import annotations
import numpy as np

from Tensor import Tensor
from Node import Node
from debug import DEBUG


class Layer:
    global_id = 0
    def __init__(self, nodes: list[Node], bias: Tensor):
        self.weights_matrix = Tensor(np.array([n.weights.data for n in nodes]), requires_gradient=True)
        self.bias = bias
        self._id = Layer.global_id
        Layer.global_id += 1