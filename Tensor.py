from __future__ import annotations
import numpy as np
from debug import DEBUG

class Tensor:
    global_id = 0
    
    def __init__(self, data: np.array, requires_gradient: bool):
        self.data = data
        self.shape = data.shape
        self.requires_gradient = requires_gradient
        self._parents = []
        self._backward = None
        self.grad = None
        self._id = Tensor.global_id
        Tensor.global_id += 1
    
    # Helper function to set the parents of this Tensor object
    # given its parent Tensors
    def set_parents(self, parents: list[Tensor]):
        self._parents = parents
        
    def backward(self):
        # To start the recursive process
        self.grad = np.ones_like(self.data)
        
        topological_order = []
        visited = set()
        
        def postOrderDFS(tensor: Tensor):
            if tensor is None or tensor in visited:
                return
            
            visited.add(tensor)
            
            for parent_tensor in tensor._parents:
                postOrderDFS(parent_tensor)
                
            topological_order.append(tensor)
            
        postOrderDFS(self)
        
        if DEBUG:
            print("Forward Topological Ordering (Parent -> Child):")
            for tensor in topological_order:
                print(tensor._id)
        
        for tensor in reversed(topological_order):
            if tensor._backward is not None:
                tensor._backward()
        
        
    # Assuming we will only be working with 1D Tensors at the moment
    # i.e. only working with scalars until everything is working
    def __add__(self, other_parent: Tensor):
        # Adding two numpy arrays together to produce another numpy array
        child_data = self.data + other_parent.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} + {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += child.grad
            else:
                self.grad = child.grad
                            
            if other_parent.grad is not None:
                other_parent.grad += child.grad
            else:
                other_parent.grad = child.grad
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                            
        child._backward = _backward
        
        return child

        
    def __sub__(self, other_parent: Tensor):
        # Subtracting two numpy arrays together to produce another numpy array
        child_data = self.data - other_parent.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} - {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += child.grad
            else:
                self.grad = child.grad
                            
            if other_parent.grad is not None:
                other_parent.grad -= child.grad
            else:
                other_parent.grad = -child.grad
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def __mul__(self, other_parent: Tensor):
        # Multiplying two numpy arrays together to produce another numpy array
        child_data = self.data * other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} * {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += (other_parent.data) * child.grad
            else:
                self.grad = (other_parent.data) * child.grad
                            
            if other_parent.grad is not None:
                other_parent.grad += (self.data) * child.grad
            else:
                other_parent.grad = (self.data) * child.grad
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def __matmul__(self, other_parent: Tensor):
        # Mat Multiplying two numpy arrays together to produce another numpy array
        child_data = self.data @ other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} @ {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += child.grad @ other_parent.data.T
            else:
                self.grad = child.grad @ other_parent.data.T
                            
            if other_parent.grad is not None:
                other_parent.grad += self.data.T @ child.grad
            else:
                other_parent.grad = self.data.T @ child.grad
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def __truediv__(self, other_parent: Tensor):
        # Dividing two numpy arrays together to produce another numpy array
        child_data = self.data / other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} / {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += child.grad / other_parent.data
            else:
                self.grad = child.grad / other_parent.data
                            
            if other_parent.grad is not None:
                other_parent.grad -= (child.grad * self.data) / (other_parent.data**2)
            else:
                other_parent.grad = -(child.grad * self.data) / (other_parent.data**2)
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def exp(parent: Tensor):
        # The raw tensor data, exponentiated
        child_data = np.exp(parent.data)

        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: exp({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if parent.grad is not None:
                parent.grad += (child.grad * np.exp(parent.data))
            else:
                parent.grad = (child.grad * np.exp(parent.data))
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def sin(parent: Tensor):
        # The raw tensor data, with sin applied
        child_data = np.sin(parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: sin({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if parent.grad is not None:
                parent.grad += (child.grad * np.cos(parent.data))
            else:
                parent.grad = (child.grad * np.cos(parent.data))
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def cos(parent: Tensor):
        # The raw tensor data, with cos applied
        child_data = np.cos(parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: cos({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if parent.grad is not None:
                parent.grad += (child.grad * -np.sin(parent.data))
            else:
                parent.grad = (child.grad * -np.sin(parent.data))
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
    def transpose(self):
        # The raw tensor data, with cos applied
        child_data = self.data.T

        # If parent of output requires gradient, child requires gradient
        if self.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [self]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: tranpose({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is not None:
                self.grad += child.grad.T
            else:
                self.grad = child.grad.T
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def __getitem__(self, idx):
        # The raw tensor data, indexed at idx
        child_data = self.data[idx]

        # If parent of output requires gradient, child requires gradient
        if self.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(np.array(child_data), requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [self]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id}[{idx}]")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            self.grad[idx] += child.grad
                
            if DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
        
    
    