import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Node import Node
from sklearn.model_selection import train_test_split

# f(x) = 2x + 1
def f(inputs: Tensor):
    return 2*inputs.data[0] + 1

def J(y_preds: Tensor, y_true: Tensor):
    n = len(y_preds)
    return (y_preds - y_true) * (y_preds - y_true) / n


def main():
    # Build data distribution
    X = np.linspace(0, 50, 100)
    y = f(X)
    
    # Split data into test, train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    
    # Build initial model
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
    
    # Run gradient descent
    num_epochs = 100
    batch_size = 1 # we will come back to making this larger
    num_batches = len(x_train) / batch_size
    learning_rate = 0.01
    
    # For each epoch (for each run through the training data)
    for _ in range(num_epochs):
        # For each batch
        for x_batch, y_batch in batches(x_train, y_train):
            # Calculate forward pass
            z = network.forward(x_batch)
            
        

def batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    

if __name__=="__main__":
    main()