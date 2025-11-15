import numpy as np
from Tensor import Tensor

# Any arbitrary function will be a function of an arbitrary number of Tensor objects
def f(input: Tensor):
    return Tensor.sin(input[0][0]) + Tensor.exp(input[0][0]*input[0][1]) - Tensor.cos(input[0][1])


# Example usage
def main():
    # Pass in input as a numpy array
    input_tensor = Tensor(np.array([[1,2]]), requires_gradient=True)
    
    z = f(input_tensor)
    
    # Get gradient of f with respect to inputs x_1, x_2
    print(z.data)
    z.backward()
    
    print(input_tensor.grad)
    
    
    
if __name__=="__main__":
    main()