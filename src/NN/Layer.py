from __future__ import annotations
import numpy as np

from autodiff import Tensor
from utils import DEBUG


class Layer:
    global_id = 0
    def __init__(self, weights_tensor: Tensor, bias_tensor: Tensor, activation: callable):
        self.weights_tensor = weights_tensor
        self.bias_tensor = bias_tensor
        self.activation = activation
        self._id = Layer.global_id
        Layer.global_id += 1
    