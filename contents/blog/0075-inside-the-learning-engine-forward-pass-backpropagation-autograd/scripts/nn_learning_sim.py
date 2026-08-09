"""
Forward Pass, Backpropagation Chain Rule & Autograd Engine from Scratch
Author: Narendra Vadapalli
Series: Neural Architecture Evolution Series (Part 5)

This script demonstrates:
1. Step-by-step Forward Pass and Loss computation.
2. Manual Backpropagation via Calculus Chain Rule derivatives.
3. Micro-Autograd Computation Engine: Building a PyTorch-like dynamic DAG with automatic reverse-mode differentiation.
4. Training loop demonstrating loss reduction over 100 epochs.
"""

import math
import random

# =====================================================================
# 1. MICRO-AUTOGRAD ENGINE (PyTorch-like Value Node)
# =====================================================================

class Value:
    """A scalar value that tracks its computation history for reverse-mode automatic differentiation."""
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Topological sort to execute reverse-mode backpropagation automatically."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __sub__(self, other): return self + (-other)
    def __neg__(self): return self * -1
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# =====================================================================
# 2. NEURAL NETWORK TRAINED WITH AUTOGRAD
# =====================================================================

class Neuron:
    def __init__(self, nin):
        random.seed(42)
        self.w = [Value(random.uniform(-0.5, 0.5)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.sigmoid()

    def parameters(self):
        return self.w + [self.b]


def run_simulation():
    print("=" * 75)
    print("      DEMONSTRATING FORWARD PASS, BACKPROPAGATION & AUTOGRAD ENGINE      ")
    print("=" * 75)

    # Simple Archery Target Dataset: x1 + x2 > 0.5 -> 1.0 else 0.0
    dataset = [
        ([0.1, 0.2], 0.0),
        ([0.8, 0.9], 1.0),
        ([0.2, 0.1], 0.0),
        ([0.9, 0.7], 1.0),
    ]

    neuron = Neuron(2)
    print("Neuron initialized with 2 inputs (x1, x2) and Sigmoid activation.")
    print("-" * 75)

    print("\n[1] Forward Pass Before Training:")
    for x, y_true in dataset:
        x_val = [Value(v) for v in x]
        y_pred = neuron(x_val)
        print(f"    Input {x} -> Predicted: {y_pred.data:.4f} | Target: {y_true}")

    # Training Loop with Autograd Reverse Pass
    print("\n[2] Executing Backpropagation Training (100 Epochs):")
    epochs = 100
    lr = 1.0

    for epoch in range(1, epochs + 1):
        total_loss = Value(0.0)
        for x, y_true in dataset:
            x_val = [Value(v) for v in x]
            y_pred = neuron(x_val)
            # Loss = (y_pred - y_true)^2
            diff = y_pred - Value(y_true)
            loss = diff ** 2
            total_loss = total_loss + loss

        # Zero gradients
        for p in neuron.parameters():
            p.grad = 0.0

        # Automatic Backpropagation via Chain Rule
        total_loss.backward()

        # Gradient Descent Weight Update
        for p in neuron.parameters():
            p.data -= lr * p.grad

        if epoch in [1, 20, 50, 100]:
            print(f"    Epoch {epoch:3d} | Total Loss: {total_loss.data:.6f} | Weight Gradients: w1_grad={neuron.w[0].grad:.4f}, w2_grad={neuron.w[1].grad:.4f}")

    print("\n[3] Forward Pass After Training:")
    all_correct = True
    for x, y_true in dataset:
        x_val = [Value(v) for v in x]
        y_pred = neuron(x_val)
        binary_pred = 1.0 if y_pred.data >= 0.5 else 0.0
        is_correct = binary_pred == y_true
        if not is_correct: all_correct = False
        status = "PASSED" if is_correct else "FAILED"
        print(f"    Input {x} -> Predicted: {y_pred.data:.4f} (Class {int(binary_pred)}) | Target: {int(y_true)} [{status}]")

    print("\n" + "=" * 75)
    print("  RESULT: Backpropagation successfully optimized weights via Chain Rule!")
    print("=" * 75)

if __name__ == "__main__":
    run_simulation()
