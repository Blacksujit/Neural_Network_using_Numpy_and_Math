import numpy as np

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Initialize weights and biases
def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(1)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))
    
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Compute loss (binary cross-entropy)
def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return loss

# Backward propagation
def backward_propagation(X, Y, cache, W1, W2):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Train the neural network
def train(X, Y, input_dim, hidden_dim, output_dim, iterations, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)
    
    for i in range(iterations):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, cache, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    return W1, b1, W2, b2

# Make predictions
def predict(X, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = (A2 > 0.5).astype(int)
    return predictions
