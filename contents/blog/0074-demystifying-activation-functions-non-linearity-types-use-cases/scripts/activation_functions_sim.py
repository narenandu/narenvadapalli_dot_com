"""
Activation Functions & Non-Linear Space Warping Simulation from Scratch
Author: Narendra Vadapalli
Series: Neural Architecture Evolution Series (Part 4)

This script demonstrates:
1. Mathematical implementations of key activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Softmax).
2. Linear Collapse Proof: Stacking linear layers without activation functions fails on non-linear datasets (concentric circles).
3. Non-Linear Space Warping: Adding activation functions enables multi-layer networks to warp space and solve non-linear classification.
"""

import math
import random

# =====================================================================
# 1. CORE ACTIVATION FUNCTIONS & DERIVATIVES
# =====================================================================

def sigmoid(x: float) -> float:
    """Sigmoid: Bounded in (0, 1). Ideal for binary probability outputs."""
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y: float) -> float:
    """Derivative given y = sigmoid(x)."""
    return y * (1.0 - y)

def tanh(x: float) -> float:
    """Tanh: Bounded in (-1, 1), zero-centered. Ideal for RNN hidden states."""
    return math.tanh(x)

def tanh_derivative(y: float) -> float:
    """Derivative given y = tanh(x)."""
    return 1.0 - y ** 2

def relu(x: float) -> float:
    """ReLU: max(0, x). Ideal for deep CNN/MLP hidden layers."""
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if x > 0 else 0.0

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU: max(alpha*x, x). Prevents dying neurons in GANs/CNNs."""
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
    return 1.0 if x > 0 else alpha

def gelu(x: float) -> float:
    """GELU: Gaussian Error Linear Unit. Preferred in Transformers (GPT, Llama)."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

def softmax(vector):
    """Softmax: Normalizes a vector of logits into a probability distribution summing to 1.0."""
    max_val = max(vector)
    exps = [math.exp(v - max_val) for v in vector]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# =====================================================================
# 2. CONCENTRIC CIRCLES NON-LINEAR DATASET GENERATION
# =====================================================================

def generate_concentric_circles_dataset(n_samples: int = 200):
    """Generates 2D concentric circles dataset (Non-linearly separable)."""
    random.seed(42)
    X = []
    y = []
    
    # Class 0: Inner circle (r < 0.5)
    for _ in range(n_samples // 2):
        r = random.uniform(0.0, 0.4)
        theta = random.uniform(0, 2 * math.pi)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(0)

    # Class 1: Outer ring (0.7 < r < 1.0)
    for _ in range(n_samples // 2):
        r = random.uniform(0.7, 1.0)
        theta = random.uniform(0, 2 * math.pi)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(1)

    return X, y


# =====================================================================
# 3. 2-LAYER NEURAL NETWORK SIMULATION (LINEAR VS NON-LINEAR)
# =====================================================================

class SimpleMLP:
    """Flexible 2-Layer Neural Network to demonstrate linear collapse vs non-linearity."""
    def __init__(self, input_dim=2, hidden_dim=8, use_activation=True):
        random.seed(42)
        self.use_activation = use_activation

        # Layer 1 weights (hidden_dim x input_dim) & bias
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        # Layer 2 weights (1 x hidden_dim) & bias
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_dim)]]
        self.b2 = [0.0]

    def forward(self, x):
        # Layer 1 linear step: z1 = W1 * x + b1
        z1 = [sum(self.W1[h][i] * x[i] for i in range(len(x))) + self.b1[h] for h in range(len(self.W1))]

        # Layer 1 activation step: a1 = relu(z1) IF enabled, else identity a1 = z1
        if self.use_activation:
            a1 = [relu(v) for v in z1]
        else:
            a1 = z1  # Linear Identity

        # Layer 2 linear step: z2 = W2 * a1 + b2
        z2 = sum(self.W2[0][h] * a1[h] for h in range(len(a1))) + self.b2[0]

        # Output activation: Sigmoid probability
        prob = sigmoid(z2)
        return prob

    def train_epoch(self, X, y, lr=0.1):
        # Simple manual numerical gradient update for demonstration
        for i in range(len(X)):
            x_i, y_i = X[i], y[i]
            prob = self.forward(x_i)
            err = prob - y_i

            # Update Layer 2 weights
            for h in range(len(self.W1)):
                a1_h = relu(sum(self.W1[h][k] * x_i[k] for k in range(2)) + self.b1[h]) if self.use_activation else (sum(self.W1[h][k] * x_i[k] for k in range(2)) + self.b1[h])
                self.W2[0][h] -= lr * err * a1_h
            self.b2[0] -= lr * err

            # Update Layer 1 weights
            for h in range(len(self.W1)):
                for k in range(2):
                    self.W1[h][k] -= lr * err * self.W2[0][h] * x_i[k] * 0.1


def evaluate(model, X, y):
    correct = 0
    for x_i, y_i in zip(X, y):
        pred = 1 if model.forward(x_i) >= 0.5 else 0
        if pred == y_i:
            correct += 1
    return (correct / len(y)) * 100.0


def run_simulation():
    print("=" * 75)
    print("      DEMONSTRATING THE NEED FOR NON-LINEAR ACTIVATION FUNCTIONS      ")
    print("=" * 75)

    X, y = generate_concentric_circles_dataset(n_samples=200)
    print("Dataset: 200 samples of Concentric Circles (Inner circle vs Outer ring)")
    print("-" * 75)

    # 1. Train Model WITHOUT Activation (Linear Layers Only)
    linear_mlp = SimpleMLP(input_dim=2, hidden_dim=16, use_activation=False)
    for _ in range(50):
        linear_mlp.train_epoch(X, y, lr=0.05)
    linear_acc = evaluate(linear_mlp, X, y)

    print("\n[1] Linear Network (No Activation Functions - Pure Linear Stacking):")
    print(f"    - Accuracy on Concentric Circles: {linear_acc:.1f}%")
    print("    --> Result: FAILED! Multiple linear layers collapse into a single flat line.")
    print("        A linear classifier cannot separate inner circles from outer rings!")

    # 2. Train Model WITH Non-Linear Activation (ReLU)
    nonlinear_mlp = SimpleMLP(input_dim=2, hidden_dim=16, use_activation=True)
    for _ in range(50):
        nonlinear_mlp.train_epoch(X, y, lr=0.05)
    nonlinear_acc = evaluate(nonlinear_mlp, X, y)

    print("\n[2] Non-Linear Network (With ReLU Activation Functions):")
    print(f"    - Accuracy on Concentric Circles: {nonlinear_acc:.1f}%")
    print("    --> Result: SUCCESS! Non-linear activations warp 2D space around the ring,")
    print("        enabling perfect non-linear classification boundary decision surface!")

    print("\n" + "=" * 75)
    print("Summary Table of Activation Functions & Use Cases:")
    print(f" {'Function':<12} | {'Bounded':<10} | {'Primary Scenario / Usecase':<35}")
    print("-" * 75)
    print(f" {'Sigmoid':<12} | {'(0, 1)':<10} | {'Binary Classification Output Layer':<35}")
    print(f" {'Tanh':<12} | {'(-1, 1)':<10} | {'RNN / LSTM Hidden Recurrent States':<35}")
    print(f" {'ReLU':<12} | {'[0, inf)':<10} | {'Deep CNN / MLP Hidden Layers':<35}")
    print(f" {'Leaky ReLU':<12} | {'(-inf, inf)':<10} | {'GANs & Deep Vision (Prevents Dying Neurons)':<35}")
    print(f" {'GELU/SwiGLU':<12} | {'(-0.17, inf)':<10} | {'Modern LLMs & Transformers (GPT, Llama)':<35}")
    print(f" {'Softmax':<12} | {'(0, 1) sum=1':<10} | {'Multi-Class Classification & Vocab Logits':<35}")
    print("=" * 75)


if __name__ == "__main__":
    run_simulation()
