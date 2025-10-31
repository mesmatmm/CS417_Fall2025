## Perceptron Learning Algorithm
## Delta Rule

import numpy as np  # Import NumPy for vector and matrix operations
import random


# Define a simple Perceptron model
class Perceptron:
    def __init__(self, input_size, epochs=10, learning_rate=0.01):
        # Initialize weights (one per input feature) to zero
        self.weights = np.zeros(input_size)
        # Initialize bias to zero
        self.bias = 0

        # Iniliaze weights and bias with random values
        # self.random_init(input_size)

        self.epochs = epochs
        self.learning_rate = learning_rate

    def random_init(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = random.random()

    @staticmethod
    def step(x):
        # Step activation function: returns 1 if x >= 0, else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Compute the linear combination of inputs and weights, add bias,
        # then apply the step activation function to get the output (0 or 1)
        return Perceptron.step(x @ self.weights + self.bias)

    def train(self, X, y):
        # Initialize an array to store predicted values
        # y_hat = np.zeros_like(y)
        # Perform up to 10 training iterations (epochs)
        for _ in range(self.epochs):
            # Iterate through all training examples
            for i, x in enumerate(X):
                # Compute the prediction for the current input
                y_hat = self.predict(x)
                # Update rule for weights and bias
                # (y[i] - y_hat[i]) gives the prediction error
                self.weights = self.weights + self.learning_rate * (y[i] - y_hat) * x
                self.bias = self.bias + self.learning_rate * (y[i] - y_hat)
            # Stop training early if all predictions are correct
            # if (y == y_hat).all():
            #     break


# Create a Perceptron for 2 input features (for AND gate)
and_p = Perceptron(2, learning_rate=1)

# Generate input data for AND gate: [[0,0], [0,1], [1,0], [1,1]]
X = [[a, b] for a in range(2) for b in range(2)]
X = np.array(X)

# Corresponding AND gate outputs: [0, 0, 0, 1]
y_and = [a and b for a, b in X]
y_and = np.array(y_and)

print(f"Before Training for AND Perceptron: weights = {and_p.weights.round(3)} and bias = {and_p.bias:3=.3F}")
# Train the perceptron using the inputs and target outputs
and_p.train(X, y_and)

# Print the learned weights and bias after training
print(f"After Training for AND Perceptron: weights = {and_p.weights.round(3)} and bias = {and_p.bias:.3F}")

# Test the perceptron on all input combinations
for i, x in enumerate(X):
    print(f"prediction: {and_p.predict(x)} actual: {y_and[i]}")
