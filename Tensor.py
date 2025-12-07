from __future__ import annotations
import numpy as np
from config import DEBUG, SUPER_DEBUG

class Tensor:
    global_id = 0
    
    def __init__(self, data: np.array, requires_gradient: bool):
        # The raw Tensor data, stored as an n-dimensional nd.array
        self.data = data
        # The shape of the Tensor, i.e. the shape of the n-dimensional nd.array
        self.shape = data.shape
        """
        Boolean denoting whether we want to calculate the partial derivative
        of the overall function with respect to this Tensor
        e.g if we have input data Tensor, this is just data (not a parameter)
        so we dont need to calculate the tensors gradient
        """
        self.requires_gradient = requires_gradient
        # The Tensors used to produce this Tensor
        self._parents = []
        # The function that propogates this Tensor's 
        # gradient to its parents
        self._backward = None
        self._backward_metadata = None
        # The gradient of the overall function 
        # with respect to this Tensor
        self.grad = None
        # id used to identify the Tensor for debugging
        self._id = Tensor.global_id
        Tensor.global_id += 1
    
    def __str__(self):
        header = f"------ Tensor {self._id} ------\n"
        data = str(self.data) + "\n"
        shape = f"Shape: {self.shape}\n"
        footer = "-------------------------------"
        return header + data + shape + footer

    
    # Helper function to set the parents of this Tensor object
    # given its parent Tensors
    def set_parents(self, parents: list[Tensor]):
        self._parents = parents
        
    def backward(self):
        # To start the recursive process
        self.grad = np.ones_like(self.data)
        # Array to store the topological ordering of
        # Tensors in the computational graph
        topological_order = []
        """
        Set used in Post-Order DFS algorithm
        to store which nodes (Tensors) we have
        already visited
        """
        visited = set()
        
        # Post-Order DFS helper function
        def postOrderDFS(tensor: Tensor):
            # If the Tensor is None (at a leaf node)
            # or we have already visited this node
            # then return
            if tensor is None or tensor in visited:
                return
            
            # Otherwise visit the Tensor
            visited.add(tensor)
            
            # For every parent of this Tensor
            for parent_tensor in tensor._parents:
                # Run Post-Order DFS on it
                postOrderDFS(parent_tensor)
            
            # After running DFS on all parents, we have ensured that all
            # dependencies of this tensor (the nodes that it depends on)
            # appear earlier in the topological order.
            # So now we can safely add this tensor itself.
            topological_order.append(tensor)
            
        postOrderDFS(self)
        
        if DEBUG:
            print("Forward Topological Ordering (Parent -> Child):")
            for tensor in topological_order:
                print(tensor._id)
        
        for tensor in reversed(topological_order):
            if tensor._backward is not None:
                tensor._backward()
    
    def as_Tensor(x):
        if isinstance(x, Tensor):
            return x
        else:
            return Tensor(np.array(x), requires_gradient=False)
                
    def _reduce_grad_for_broadcast(child_grad: np.array, parent_shape: tuple,  child_shape: tuple):
        """
        child_grad: The gradient flowing into this parent from upstream
        parent_shape: The shape of the parent before broadcasting
        child_shape: The shape of the output child after broadcasing
        (if parent_shape != child_shape ==> broadcasing has occurred)
        """
        # parent_shape will always have smaller rank than child_shape
        # rank = # of indices required to identify an element
        rank_diff = len(child_shape) - len(parent_shape)
        parent_shape_padded = (1,) * rank_diff + parent_shape
        
        axes_to_sum = []
        # Anywhere parent_shape_padded == 1 and  child_shape != 1, a broadcast took place
        for i, (in_dim, out_dim) in enumerate(zip(parent_shape_padded, child_shape)):
            if in_dim == 1 and out_dim != 1:
                axes_to_sum.append(i)
                
        if axes_to_sum:
            child_grad = child_grad.sum(axis=tuple(axes_to_sum), keepdims=False)
                
        return child_grad.reshape(parent_shape)
        
    # Assuming we will only be working with 1D Tensors at the moment
    # i.e. only working with scalars until everything is working
    def __add__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # Adding two numpy arrays together to produce another numpy array
        child_data = self.data + other_parent.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} + {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            child_grad_self = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=self.shape,
                                                          child_shape=child.grad.shape)
            if self.grad is not None:
                self.grad += child_grad_self
            else:
                self.grad = child_grad_self
                            
            child_grad_other = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=other_parent.shape,
                                                          child_shape=child.grad.shape)
            if other_parent.grad is not None:
                other_parent.grad += child_grad_other
            else:
                other_parent.grad = child_grad_other
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                            
        child._backward = _backward
        
        return child

        
    def __sub__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # Subtracting two numpy arrays together to produce another numpy array
        child_data = self.data - other_parent.data
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        # The new child Tensor
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} - {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
                
            child_grad_self = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=self.shape,
                                                          child_shape=child.grad.shape)
            if self.grad is not None:
                self.grad += child_grad_self
            else:
                self.grad = child_grad_self
            
            child_grad_other = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=other_parent.shape,
                                                          child_shape=child.grad.shape)
            if other_parent.grad is not None:
                other_parent.grad -= child_grad_other
            else:
                other_parent.grad = -child_grad_other
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def __mul__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # Multiplying two numpy arrays together to produce another numpy array
        child_data = self.data * other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} * {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
                
            child_grad_self = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=self.shape,
                                                          child_shape=child.grad.shape)
            if self.grad is not None:
                self.grad += (other_parent.data) * child_grad_self
            else:
                self.grad = (other_parent.data) * child_grad_self
            
            child_grad_other = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=other_parent.shape,
                                                          child_shape=child.grad.shape)
            if other_parent.grad is not None:
                other_parent.grad += (self.data) * child_grad_other
            else:
                other_parent.grad = (self.data) * child_grad_other
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def __matmul__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # Mat Multiplying two numpy arrays together to produce another numpy array
        child_data = self.data @ other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(child_data, requires_gradient=requires_gradient)
        
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
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def __truediv__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # Dividing two numpy arrays together to produce another numpy array
        child_data = self.data / other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} / {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
                
            child_grad_self = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=self.shape,
                                                          child_shape=child.grad.shape)
            if self.grad is not None:
                self.grad += child_grad_self / other_parent.data
            else:
                self.grad = child_grad_self / other_parent.data

            child_grad_other = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=other_parent.shape,
                                                          child_shape=child.grad.shape)
            if other_parent.grad is not None:
                other_parent.grad -= (child_grad_other * self.data) / (other_parent.data**2)
            else:
                other_parent.grad = -(child_grad_other * self.data) / (other_parent.data**2)
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def __pow__(self, other_parent: Tensor):
        # Make sure any input is a Tensor
        other_parent = Tensor.as_Tensor(other_parent)
        # exponentiating one numpy array by the parent to produce another numpy array
        child_data = self.data ** other_parent.data
        
        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if self.requires_gradient or other_parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False

        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parents
        parents = [self, other_parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id} ** {parents[1]._id}")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            child_grad_self = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=self.shape,
                                                          child_shape=child.grad.shape)
            if self.grad is not None:
                self.grad += child_grad_self * other_parent.data * (self.data ** (other_parent.data -  1))
            else:
                self.grad = child_grad_self * other_parent.data * (self.data ** (other_parent.data -  1))
            
            child_grad_other = Tensor._reduce_grad_for_broadcast(child_grad=child.grad, 
                                                          parent_shape=other_parent.shape,
                                                          child_shape=child.grad.shape)
            if other_parent.grad is not None:
                other_parent.grad += child_grad_other * (self.data ** other_parent.data) * np.log(self.data)
            else:
                other_parent.grad = child_grad_other * (self.data ** other_parent.data) * np.log(self.data)
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
        
        
    def exp(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, exponentiated
        child_data = np.exp(parent.data)

        # If either parent of the output Tensor requires a gradient
        # Then this output Tensor will also require a gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
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
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def sin(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, with sin applied
        child_data = np.sin(parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
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
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def cos(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, with cos applied
        child_data = np.cos(parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
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
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def sigmoid(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, with sigmoid applied
        child_data = 1 / (1 + np.exp(-parent.data))

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: sigmoid({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if parent.grad is not None:
                # Eventually look into caching this value during the forward pass
                parent.grad += child.grad * child.data * (1 - child.data)
            else:
                parent.grad = child.grad * child.data * (1 - child.data)
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def ReLU(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, with ReLU applied
        child_data = np.maximum(0, parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: ReLU({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
                
            grad_mask = (parent.data > 0).astype(float)
            if parent.grad is not None:
                parent.grad += child.grad * grad_mask
            else:
                parent.grad = child.grad * grad_mask
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    def tanh(parent: Tensor):
        # Make sure any input is a Tensor
        parent = Tensor.as_Tensor(parent)
        # The raw tensor data, with tanh applied
        child_data = np.tanh(parent.data)

        # If parent of output requires gradient, child requires gradient
        if parent.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
        # Setting the new child Tensor's parent
        parents = [parent]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: tanh({parents[0]._id})")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if parent.grad is not None:
                # Eventually look into caching this value during the forward pass
                parent.grad += child.grad * child.data**2
            else:
                parent.grad = child.grad * child.data**2
                
            if SUPER_DEBUG:
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
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
        
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
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
    
    def sum(self, axis: None | int | tuple):
        # Raw numpy data summed up
        child_data = self.data.sum(axis=axis)

        # If parent of output requires gradient, child requires gradient
        if self.requires_gradient:
            requires_gradient = True
        else:
            requires_gradient = False
        
        child = Tensor(child_data, requires_gradient=requires_gradient)
                
        # Setting the new child Tensor's parent
        parents = [self]
        child.set_parents(parents)
        
        if DEBUG:
            print(f"Child: {child._id} produced from: {parents[0]._id}.sum()")
        
        # Setting the child Tensor's backward prop rule
        def _backward():
            if DEBUG: 
                print(f"Propogating gradient from {child._id} to {[parent._id for parent in child._parents]} ")
            
            if axis == None:
                if self.grad is not None:
                    self.grad += np.ones_like(self.data) * child.grad
                else:
                    self.grad = np.ones_like(self.data) * child.grad
            else:
                if isinstance(axis, int):
                    axis_tuple = (axis,)
                else:
                    axis_tuple = axis
                child_grad_reshaped = child.grad
                for axs in sorted(axis_tuple):
                    child_grad_reshaped = np.expand_dims(child_grad_reshaped, axis=axs)
                child_grad_broadcasted = np.broadcast_to(child_grad_reshaped, self.shape)
                
                if self.grad is not None:
                    self.grad += child_grad_broadcasted
                else:
                    self.grad = child_grad_broadcasted
                
                
            if SUPER_DEBUG:
                print(f"New parent gradients: {[p.grad for p in parents]}")
                                
        child._backward = _backward
        
        return child
    
        
    
    