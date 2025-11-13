import numpy as np
from Tensor import Tensor

# Any arbitrary function will be a function of an arbitrary number of Tensor objects
def f1(x, y):
    return (x*y) + Tensor.exp(x*y)

def f2(x, y):
    return Tensor.sin(x) + Tensor.exp(x*y) - Tensor.cos(y)


# Example usage
def main():
    # Pass in input as a numpy array
    input = np.array([1,2])
    # Extract parameters as Tensors, noting that we want the partial derivatives of these parameters
    x_1 = Tensor(input[0], requires_gradient=True)
    x_2 = Tensor(input[1], requires_gradient=True)
    
    # Call f with x_1 Tensor and x_2 Tensor to produce a new Tensor object, z
    # such that z.data contains the output of f
    # z will also contain the full computational graph
    z = f2(x_1, x_2)
    
    # Get gradient of f with respect to inputs x_1, x_2
    z.backward()
    
    print([x_1.grad, x_2.grad])
    
    
    
    
    
    
    
    
if __name__=="__main__":
    main()