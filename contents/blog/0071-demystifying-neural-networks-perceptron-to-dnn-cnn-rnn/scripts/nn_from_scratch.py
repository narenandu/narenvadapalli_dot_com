#!/usr/bin/env python3
"""
Zero-Dependency Python Implementation of a Multi-Layer Perceptron (MLP) from Scratch.

This script demonstrates forward propagation, backpropagation using the chain rule,
and gradient descent updates on the non-linearly separable XOR dataset without external libraries.
"""

import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output):
    # Derivation: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    return output * (1.0 - output)

class NeuralNetworkFromScratch:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, seed=42):
        random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights with small random values (Xavier/Glorot-like bounds)
        bound_h = math.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = [[random.uniform(-bound_h, bound_h) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        
        bound_o = math.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = [[random.uniform(-bound_o, bound_o) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(output_dim)]

    def forward(self, x):
        # Hidden layer forward pass: h = sigmoid(x @ W1 + b1)
        self.hidden_input = [0.0] * self.hidden_dim
        self.hidden_output = [0.0] * self.hidden_dim
        
        for j in range(self.hidden_dim):
            z1 = sum(x[i] * self.W1[i][j] for i in range(self.input_dim)) + self.b1[j]
            self.hidden_input[j] = z1
            self.hidden_output[j] = sigmoid(z1)
            
        # Output layer forward pass: y_hat = sigmoid(hidden_output @ W2 + b2)
        self.final_input = [0.0] * self.output_dim
        self.final_output = [0.0] * self.output_dim
        
        for k in range(self.output_dim):
            z2 = sum(self.hidden_output[j] * self.W2[j][k] for j in range(self.hidden_dim)) + self.b2[k]
            self.final_input[k] = z2
            self.final_output[k] = sigmoid(z2)
            
        return self.final_output

    def backward(self, x, target, lr=0.5):
        # 1. Output layer error gradient delta_k = (y_hat - target) * sigmoid_derivative(y_hat)
        output_deltas = [0.0] * self.output_dim
        for k in range(self.output_dim):
            error = self.final_output[k] - target[k]
            output_deltas[k] = error * sigmoid_derivative(self.final_output[k])

        # 2. Hidden layer error gradient delta_j = sum(delta_k * W2[j][k]) * sigmoid_derivative(h_j)
        hidden_deltas = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            error_sum = sum(output_deltas[k] * self.W2[j][k] for k in range(self.output_dim))
            hidden_deltas[j] = error_sum * sigmoid_derivative(self.hidden_output[j])

        # 3. Update W2 and b2 (hidden-to-output weights)
        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                self.W2[j][k] -= lr * output_deltas[k] * self.hidden_output[j]
        for k in range(self.output_dim):
            self.b2[k] -= lr * output_deltas[k]

        # 4. Update W1 and b1 (input-to-hidden weights)
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= lr * hidden_deltas[j] * x[i]
        for j in range(self.hidden_dim):
            self.b1[j] -= lr * hidden_deltas[j]

def run_simulation():
    # XOR Dataset: Non-linearly separable truth table
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    nn = NeuralNetworkFromScratch(input_dim=2, hidden_dim=4, output_dim=1, seed=123)
    epochs = 10000
    learning_rate = 0.5

    print("=========================================================")
    print(" Training 2-Layer Neural Network on XOR Problem (Scratch) ")
    print("=========================================================")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, target in dataset:
            prediction = nn.forward(x)
            total_loss += sum((target[k] - prediction[k]) ** 2 for k in range(len(target)))
            nn.backward(x, target, lr=learning_rate)
        
        mse_loss = total_loss / len(dataset)
        if epoch == 1 or epoch % 2000 == 0 or epoch == epochs:
            print(f"Epoch {epoch:5d} / {epochs} | Mean Squared Error Loss: {mse_loss:.6f}")

    print("\n---------------------------------------------------------")
    print(" Verification & Final Predictions on Test Set: ")
    print("---------------------------------------------------------")
    print(" Input (x1, x2) | Target | Model Output (y_hat) | Rounded Class ")
    print("---------------------------------------------------------")
    all_correct = True
    for x, target in dataset:
        pred = nn.forward(x)[0]
        binary_pred = 1.0 if pred >= 0.5 else 0.0
        is_match = (binary_pred == target[0])
        if not is_match:
            all_correct = False
        print(f"   ({int(x[0])}, {int(x[1])})     |   {int(target[0])}    |        {pred:.6f}       |       {int(binary_pred)} ")

    print("---------------------------------------------------------")
    if all_correct:
        print(" SUCCESS: MLP successfully learned the non-linear XOR function!")
    else:
        print(" FAILED: Model did not converge.")

if __name__ == "__main__":
    run_simulation()
