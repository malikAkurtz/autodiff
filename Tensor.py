import numpy as np
from __future__ import annotations
class Tensor:
    requires_gradient = None
    data = None
    _parents = None    
    _backward = None
    grad = None
    
    def __init__(self, data: np.array, requires_gradient: bool):
        self.data = data
        self.requires_gradient = requires_gradient
    
    # Helper function to set the parents of this Tensor object
    # given its parent Tensors
    def set_parents(self, parents: list[Tensor]):
        self._parents = parents
        
    def grad(self):
        self.grad = np.array([1])
        
        
        
    # Assuming we will only be working with 1D Tensors at the moment
    # i.e. only working with scalars until everything is working
    def __add__(self, other: Tensor):
        # Adding two numpy arrays together to produce another numpy array
        child_data = self.data + other.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        child.set_parents([self, other])
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if self.grad is not None:
                self.grad += child.grad
            else:
                self.grad = child.grad
            
            if other.grad is not None:
                other.grad += child.grad
            else:
                other.grad = child.grad
                
        child._backward = _backward
        
        return child

        
    def __sub__(self, other: Tensor):
        # Subtracting two numpy arrays together to produce another numpy array
        child_data = self.data - other.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        child.set_parents([self, other])
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if self.grad is not None:
                self.grad += child.grad
            else:
                self.grad = child.grad
            
            if other.grad is not None:
                other.grad -= child.grad
            else:
                other.grad = -child.grad
                
        child._backward = _backward
        
        return child
        
        
    def __mul__(self, other: Tensor):
        # Multiplying two numpy arrays together to produce another numpy array
        child_data = self.data * other.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        child.set_parents([self, other])
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if self.grad is not None:
                self.grad += (other.data) * child.grad
            else:
                self.grad = (other.data) * child.grad
            
            if other.grad is not None:
                other.grad += (self.data) * child.grad
            else:
                other.grad = (self.data) * child.grad
                
        child._backward = _backward
        
        return child
        
        
    def __truediv__(self, other: Tensor):
        # Dividing two numpy arrays together to produce another numpy array
        child_data = self.data / other.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        child.set_parents([self, other])
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if self.grad is not None:
                self.grad += child.grad / other.data
            else:
                self.grad = child.grad / other.data
            
            if other.grad is not None:
                other.grad -= (child.grad * self.data) / (other.data**2)
            else:
                other.grad = -(child.grad * self.data) / (other.data**2)
                
        child._backward = _backward
        
        return child
        
        
    def exp(tensor: Tensor):
        # The raw tensor data, exponentiated
        child_data = np.exp(tensor.data)

        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if tensor.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        child.set_parents([tensor])
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if tensor.grad is not None:
                tensor.grad += (child.grad * np.exp(tensor.data))
            else:
                tensor.grad = (child.grad * np.exp(tensor.data))
                
        child._backward = _backward
        
        return child
        
        
    
        
    
    