import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Node import Node
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# f(x) = 2x + 1
def f(x):
    return 2*x + 1

def cost_fn(y_preds: Tensor, y_true: Tensor):
    n = len(y_preds.data)
    return ((y_preds - y_true) ** 2) / n


def main():
    # Build data distribution
    X = np.linspace(0, 50, 100)
    y = f(X)
    
    # Split data into test, train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Build initial model
    L0_weights = np.array([
        [0.1, 0.3, 0.5]
    ])
    L0_bias = np.array([1.0, 1.0, 1.0])
    
    L0_weights_tensor = Tensor(L0_weights, requires_gradient=True)
    L0_bias_tensor = Tensor(L0_bias, requires_gradient=True)
    L0 = Layer(weights=L0_weights_tensor, bias=L0_bias_tensor, activation=Tensor.sigmoid)
    
    L1_weights = np.array([
        [0.7], 
        [0.8], 
        [0.9]
    ])
    L1_bias = np.array([1.0])
    
    L1_weights_tensor = Tensor(L1_weights, requires_gradient=True)
    L1_bias_tensor = Tensor(L1_bias, requires_gradient=True)
    L1 = Layer(weights=L1_weights_tensor, bias=L1_bias_tensor, activation=None)
    
    network = NeuralNetwork([L0, L1])
    
    # Run gradient descent
    num_epochs = 100
    batch_size = 16 
    learning_rate = 0.01
    
    history = {"train_loss": []}
    
    # For each epoch (for each run through the training data)
    for _ in range(num_epochs):
        # For each batch
        epoch_avg_loss = 0
        for x_batch, y_batch in batches(x_train, y_train, batch_size):
            # Perform forward pass
            x_batch_tensor = Tensor(x_batch.reshape(-1, 1), requires_gradient=False)
            y_batch_tensor = Tensor(y_batch.reshape(-1, 1), requires_gradient=False)
            batch_output = network.forward(x_batch_tensor)
            
            # Calculate cost
            batch_loss = cost_fn(batch_output, y_batch_tensor).sum()
            print(f"Epochs {_} Batch Loss: {batch_loss.data}")
            epoch_avg_loss += batch_loss.data
            
            # Propogate gradient of loss wrt to parameters
            batch_loss.backward()
            
            # Take GD step
            for layer in network.layers:
                layer.weights.data -= learning_rate * layer.weights.grad
                layer.bias.data    -= learning_rate * layer.bias.grad
                
            # Zero gradients
            for layer in network.layers:
                layer.weights.grad = np.zeros_like(layer.weights.data)
                layer.bias.grad = np.zeros_like(layer.bias.data)
                
        epoch_avg_loss /= batch_size
        history["train_loss"].append(epoch_avg_loss)
        
    # Run inference on test data
    
    y_preds = network.forward(x_test)
    test_loss = cost_fn(y_preds, y_test)
    
    print(f"Loss on Test Data: {test_loss}")
    plot_history(history, "Training Loss")
    plot_data(x_test, y_test, y_preds)
        
        
def plot_history(history, plot_title='', save_path=None):
    epochs = range(len(history["train_loss"]))
    plt.plot(epochs, history["train_loss"], color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_data(x_test, y_test, y_preds):
    plt.scatter(x_test, y_test, color="blue")
    plt.scatter(x_test, y_preds, color="red")
    plt.xlabel("X Value")
    plt.ylabel("Y Value")
    plt.legend()
    plt.grid(True)
    plt.show()
            
def batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    

if __name__=="__main__":
    main()