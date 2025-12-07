import numpy as np
from Tensor import Tensor
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE

def cost_fn(y_preds_tensor: Tensor, y_true_tensor: Tensor):
    num_elements = y_preds_tensor.shape[0]
    return ((y_preds_tensor - y_true_tensor) ** 2).sum(axis=None) / num_elements


def main():
    # Build data distribution
    X = np.linspace(0, 10, 100)
    X = X / 10
    y = 2 * X + 5
    
    # Split data into test, train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Build initial model
    L0_weights = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ])
    L0_bias = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    
    L0_weights_tensor = Tensor(L0_weights, requires_gradient=True)
    L0_bias_tensor = Tensor(L0_bias, requires_gradient=True)
    L0 = Layer(weights_tensor=L0_weights_tensor, bias_tensor=L0_bias_tensor, activation=Tensor.ReLU)
    
    L1_weights = np.array([
        [0.1], 
        [0.1], 
        [0.1],
        [0.1],
        [0.1],
        [0.1]
    ])
    L1_bias = np.array([1.0])
    
    L1_weights_tensor = Tensor(L1_weights, requires_gradient=True)
    L1_bias_tensor = Tensor(L1_bias, requires_gradient=True)
    L1 = Layer(weights_tensor=L1_weights_tensor, bias_tensor=L1_bias_tensor, activation=None)
    
    network = NeuralNetwork([L0, L1])
    
    # Run gradient descent
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE 
    learning_rate = LEARNING_RATE
    
    history = {"train_loss": []}
    
    # For each epoch (for each run through the training data)
    for _ in range(num_epochs):
        # For each batch
        epoch_avg_loss = 0
        batch_counter = 0
        for x_batch, y_batch in batches(x_train, y_train, batch_size):
            # x_batch has shape (batch_size,)
            # need to reshape to (batch_size, 1)
            x_batch = x_batch.reshape(-1, 1)
            # Cast to Tensor
            x_batch_tensor = Tensor(x_batch, requires_gradient=False)
            print("x_batch:")
            print(x_batch_tensor)
            
            # y_batch has shape(batch_size,)
            # need to reshape to (batch_size, 1)
            y_batch = y_batch.reshape(-1, 1)
            # Cast to Tensor
            y_batch_tensor = Tensor(y_batch, requires_gradient=False)
            print(f"y_batch:")
            print(y_batch_tensor)
            
            # Perform forward pass to create computational graph
            batch_output_tensor = network.forward(x_batch_tensor)
            print(f"Batch output:")
            print(batch_output_tensor)
            
            # Calculate cost
            batch_loss = cost_fn(batch_output_tensor, y_batch_tensor)
            print(f"Epoch {_} Batch {batch_counter} Loss: {batch_loss.data}")
            batch_counter += 1
            epoch_avg_loss += batch_loss.data
            
            # Perform backward pass
            batch_loss.backward()
            
            # Take GD step
            for layer in network.layers:
                layer.weights_tensor.data -= learning_rate * layer.weights_tensor.grad
                layer.bias_tensor.data    -= learning_rate * layer.bias_tensor.grad
                
            # Zero gradients
            for layer in network.layers:
                layer.weights_tensor.grad = np.zeros_like(layer.weights_tensor.data)
                layer.bias_tensor.grad = np.zeros_like(layer.bias_tensor.data)
                
        epoch_avg_loss /= batch_size
        history["train_loss"].append(epoch_avg_loss)
        
    # Run inference on test data
    # Reshape x_test, y_test
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # Cast x_test, y_test to Tensors
    x_test_tensor = Tensor(x_test, requires_gradient=False)
    y_test_tensor = Tensor(y_test, requires_gradient=False)
    y_preds_tensor = network.forward(x_test_tensor)
    test_loss_tensor = cost_fn(y_preds_tensor, y_test_tensor)
    
    print(f"Loss on Test Data: {test_loss_tensor.data}")
    plot_history(history, "Training Loss")
    plot_data(x_test_tensor.data, y_test_tensor.data, y_preds_tensor.data)
        
        
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