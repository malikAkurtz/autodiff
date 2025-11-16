from __future__ import annotations
import numpy as np

from Tensor import Tensor
from Node import Node
from debug import DEBUG


class Layer:
    global_id = 0
    def __init__(self, weights: Tensor, bias: Tensor, activation: callable):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self._id = Layer.global_id
        Layer.global_id += 1
    