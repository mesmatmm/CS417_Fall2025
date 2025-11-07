import numpy as np


# --- Step 1: Sigmoid activation function and its derivative ---
def sigmoid(x):
    """Activation function: f(x) = 1 / (1 + e^-x)"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
    return x * (1 - x)


# --- Step 2: Training data for XOR problem ---
# Inputs (4 samples, 2 features)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target outputs (XOR truth table)
y = np.array([[0],
              [1],
              [1],
              [0]])

# --- Step 3: Initialize weights and biases randomly ---
np.random.seed(42)  # for reproducibility

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights between input and hidden layer
W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))

# Weights between hidden and output layer
W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))

# Bias terms
b1 = np.zeros((1, hidden_neurons))
# print(f"b1.shape = {b1.shape}")
b2 = np.zeros((1, output_neurons))

# --- Step 4: Learning rate ---
lr = 0.5

# --- Step 5: Training loop ---
for epoch in range(100000):
    # ---- Forward Pass ----
    hidden_input = np.dot(X, W1) + b1  # Step 1: Input → Hidden
    # print(hidden_input)
    hidden_output = sigmoid(hidden_input)  # Step 2: Activation
    # print(f"hidden output: {hidden_output}")
    # break
    final_input = np.dot(hidden_output, W2) + b2  # Step 3: Hidden → Output
    final_output = sigmoid(final_input)  # Step 4: Activation
    # print(f"final output: {final_output}")
    # break

    # ---- Compute Error ----
    error = y - final_output

    # ---- Backpropagation ----
    # Output layer delta
    d_output = error * sigmoid_derivative(final_output)

    # Hidden layer delta
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # ---- Update Weights and Biases ----
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # (optional) print loss occasionally
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Step 6: Final output after training ---
print("\nFinal predictions:")
print(final_output)
