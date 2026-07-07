# File: scripts/training_vs_inference.py
# Execution: Run in python terminal or script environment
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Neural Network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# ==========================================
# 🏋️ PHASE 1: THE TRAINING PIPELINE
# ==========================================
print("--- Starting Training Phase ---")
model.train()  # 1. Enable training mode (activates dropout, batchnorm, etc.)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Mock Input and Ground Truth target
inputs = torch.randn(1, 10)
targets = torch.randn(1, 2)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)
print(f"Calculated Training Loss: {loss.item():.4f}")

# Backward pass (Backpropagation)
optimizer.zero_grad()  # Reset existing gradients
loss.backward()        # 2. Compute gradients for all weights
optimizer.step()       # 3. Update weights using optimizer
print("Weights updated successfully.\n")


# ==========================================
# 🚀 PHASE 2: THE INFERENCE PIPELINE
# ==========================================
print("--- Starting Inference Phase ---")
model.eval()  # 1. Enable evaluation/inference mode (deactivates training logic)

# Disable gradient tracking to save compute and VRAM
with torch.no_grad():  # 2. Turn off the autograd engine entirely
    inference_input = torch.randn(1, 10)
    prediction = model(inference_input)  # 3. Forward pass only
    
print(f"Inference Prediction Output: {prediction.numpy()}")
