# File: scripts/model_weights_inspector.py
# Execution: Run in python terminal or script environment
import torch
import torch.nn as nn

# Create a neural network layer: 3 inputs mapping to 2 outputs
# This mimics a simple linear connection in a network
layer = nn.Linear(in_features=3, out_features=2)

print("=== Neural Network Layer Details ===")
print(f"Layer: {layer}")

# 1. Inspect the Weight Matrix
# PyTorch randomly initializes these weights before training
print("\n1. Weight Matrix (W):")
print(layer.weight.data)
print(f"Shape: {layer.weight.shape} (out_features x in_features)")

# 2. Inspect the Bias Vector
print("\n2. Bias Vector (b):")
print(layer.bias.data)
print(f"Shape: {layer.bias.shape} (out_features)")

# 3. Simulate a forward pass: y = Wx + b
mock_input = torch.tensor([1.0, 2.0, 3.0])
print(f"\n3. Mock Input Vector (x): {mock_input.tolist()}")

output = layer(mock_input)
print(f"Computed Output (y): {output.tolist()}")
